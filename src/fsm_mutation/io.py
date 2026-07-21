"""Parsing of FSM specification files and mutant files.

File format
-----------
A specification file contains one or more FSM blocks. Each block starts with a
header line of exactly 6 whitespace-separated integers::

    <fsm_id> <num_states> <num_transitions> <num_inputs> <num_outputs> <void>

followed by transition lines of 4 tokens::

    <source> <target> <input> <output>

Blocks are separated by blank lines. Mutant files use the same layout, where
each block is one mutant and the header's ``fsm_id`` identifies which
specification the mutant belongs to.

Representation
--------------
States and input labels are remapped to dense 0-based indices so that the
search can use flat numpy arrays. ``Spec.state_labels`` / ``Spec.input_labels``
keep the original tokens so that reported input sequences can be printed in the
original alphabet.

Mutants are stored as a *sparse delta* against the specification: each mutant
differs from the spec in only a handful of transitions (typically 1-3), so we
keep just those. This is roughly three orders of magnitude smaller than holding
a full transition table per mutant, and it is what makes the memory figures in
the results reflect the search rather than the storage scheme.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Iterator

UNDEFINED = -1


# --------------------------------------------------------------------------
# Specification
# --------------------------------------------------------------------------

@dataclass
class Spec:
    """A deterministic, possibly partial, FSM specification."""

    fsm_id: int
    n_states: int
    n_inputs: int
    n_outputs: int
    n_transitions: int

    #: next_state[s, i] -> state index, or UNDEFINED
    next_state: np.ndarray
    #: output[s, i] -> output symbol, or UNDEFINED
    output: np.ndarray

    state_labels: list = field(default_factory=list)
    input_labels: list = field(default_factory=list)

    @property
    def initial(self) -> int:
        """Initial state index.

        Defined once, here, as the numerically smallest state label. Every
        algorithm uses this; there is deliberately no second definition
        anywhere in the codebase.
        """
        return 0  # state_labels is sorted, so the smallest label is index 0

    def defined_inputs(self, s: int) -> np.ndarray:
        """Input indices that are defined at state ``s`` (partial FSM safe)."""
        return np.flatnonzero(self.next_state[s] != UNDEFINED)

    def render(self, seq) -> str:
        """Render a sequence of input indices using the original alphabet."""
        return "".join(str(self.input_labels[i]) for i in seq)


def _iter_blocks(path) -> Iterator[tuple[list[str], list[list[str]]]]:
    """Yield ``(header_tokens, transition_rows)`` for each block in a file.

    Blocks are delimited by header lines (6 tokens). Blank lines are tolerated
    but not required as separators, which makes this robust to files that do
    or do not end with a trailing newline.
    """
    header = None
    rows: list[list[str]] = []
    with open(path, "r") as fh:
        for raw in fh:
            parts = raw.split()
            if not parts:
                continue
            if len(parts) == 6 and parts[0].lstrip("-").isdigit():
                if header is not None:
                    yield header, rows
                header, rows = parts, []
            elif len(parts) >= 4 and header is not None:
                rows.append(parts[:4])
    if header is not None:
        yield header, rows


def read_specs(path) -> list[Spec]:
    """Read every specification in ``path``."""
    specs = []
    for header, rows in _iter_blocks(path):
        fsm_id = int(header[0])
        declared_states = int(header[1])
        declared_trans = int(header[2])
        declared_inputs = int(header[3])
        declared_outputs = int(header[4])

        state_labels = sorted({int(r[0]) for r in rows} | {int(r[1]) for r in rows})
        input_labels = sorted({r[2] for r in rows})
        s_index = {lab: k for k, lab in enumerate(state_labels)}
        i_index = {lab: k for k, lab in enumerate(input_labels)}

        ns = len(state_labels)
        ni = len(input_labels)
        next_state = np.full((ns, ni), UNDEFINED, dtype=np.int32)
        output = np.full((ns, ni), UNDEFINED, dtype=np.int32)

        for src, tgt, inp, out in rows:
            s, i = s_index[int(src)], i_index[inp]
            next_state[s, i] = s_index[int(tgt)]
            output[s, i] = int(out)

        specs.append(
            Spec(
                fsm_id=fsm_id,
                n_states=ns,
                n_inputs=ni,
                n_outputs=declared_outputs,
                n_transitions=len(rows),
                next_state=next_state,
                output=output,
                state_labels=state_labels,
                input_labels=input_labels,
            )
        )

        # Surface disagreements between the header and the actual content
        # rather than silently trusting the header.
        if ns != declared_states or len(rows) != declared_trans or ni != declared_inputs:
            import warnings

            warnings.warn(
                f"FSM {fsm_id}: header declares "
                f"{declared_states} states / {declared_trans} transitions / "
                f"{declared_inputs} inputs, file contains "
                f"{ns} / {len(rows)} / {ni}."
            )
    return specs


# --------------------------------------------------------------------------
# Mutants
# --------------------------------------------------------------------------

@dataclass
class MutantSet:
    """Mutants of one specification, stored as a sorted sparse delta.

    ``keys`` is a sorted int64 array of ``mutant * (n_states * n_inputs)
    + state * n_inputs + input`` for every transition where the mutant differs
    from the specification. ``delta_next`` / ``delta_out`` hold the mutant's
    values at those positions. Lookups are done with a vectorised
    ``np.searchsorted``, so an entire population steps in one operation.
    """

    n_mutants: int
    keys: np.ndarray
    delta_next: np.ndarray
    delta_out: np.ndarray
    stride: int  # n_states * n_inputs

    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.delta_next.nbytes + self.delta_out.nbytes)


def read_mutants(path, spec: Spec, limit: int | None = None) -> MutantSet:
    """Read mutants belonging to ``spec`` from ``path``.

    Only blocks whose header id matches ``spec.fsm_id`` are materialised;
    everything else is skipped without allocating.
    """
    s_index = {lab: k for k, lab in enumerate(spec.state_labels)}
    i_index = {lab: k for k, lab in enumerate(spec.input_labels)}
    stride = spec.n_states * spec.n_inputs

    keys: list[int] = []
    dnext: list[int] = []
    dout: list[int] = []
    count = 0

    for header, rows in _iter_blocks(path):
        if int(header[0]) != spec.fsm_id:
            continue
        base = count * stride
        changed = False
        for src, tgt, inp, out in rows:
            s = s_index.get(int(src))
            i = i_index.get(inp)
            if s is None or i is None:
                # Mutant references a state/input absent from the spec.
                continue
            t = s_index.get(int(tgt))
            o = int(out)
            if t is None:
                continue
            if t != spec.next_state[s, i] or o != spec.output[s, i]:
                keys.append(base + s * spec.n_inputs + i)
                dnext.append(t)
                dout.append(o)
                changed = True
        if not changed:
            # Identical to the spec: not a mutant, skip without consuming an id.
            continue
        count += 1
        if limit is not None and count >= limit:
            break

    if count == 0:
        return MutantSet(0, np.empty(0, np.int64), np.empty(0, np.int32),
                         np.empty(0, np.int32), stride)

    keys_a = np.asarray(keys, dtype=np.int64)
    order = np.argsort(keys_a, kind="stable")
    return MutantSet(
        n_mutants=count,
        keys=keys_a[order],
        delta_next=np.asarray(dnext, dtype=np.int32)[order],
        delta_out=np.asarray(dout, dtype=np.int32)[order],
        stride=stride,
    )


def write_fsm(fh, header, rows) -> None:
    """Write one FSM block followed by a blank line."""
    fh.write(" ".join(str(t) for t in header) + "\n")
    for r in rows:
        fh.write(" ".join(str(t) for t in r) + "\n")
    fh.write("\n")
