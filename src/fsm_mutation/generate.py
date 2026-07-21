"""Mutant generation.

Fault types
-----------
=====  ==========================================================
  0    output fault -- change a transition's output
  1    transfer fault -- change a transition's next state
  2    mixed -- randomly one or two faults per chosen transition
  3    transfer *and* output fault on the same transition
=====  ==========================================================

Every emitted mutant is guaranteed to be

* **distinct** -- no two mutants carry the same set of changed transitions, and
* **non-equivalent** -- distinguishable from the specification by *output*,
  verified by a breadth-first walk of the product automaton.

The equivalence check compares outputs only, which is the same criterion the
search algorithms use to declare a mutant killed. Generating under one
definition and scoring under another is what made earlier mutation scores
unreproducible.
"""

from __future__ import annotations

import random
from collections import deque

from .io import UNDEFINED, Spec, _iter_blocks, write_fsm


def is_distinguishable(spec: Spec, delta: dict) -> bool:
    """True if the mutant given by ``delta`` differs from ``spec`` in output.

    ``delta`` maps ``(state, input)`` to ``(next_state, output)``. The walk
    explores pairs of (specification state, mutant state) reachable under the
    same input sequence and reports the first output disagreement.
    """
    start = (spec.initial, spec.initial)
    seen = {start}
    queue = deque([start])

    while queue:
        s, m = queue.popleft()
        for i in range(spec.n_inputs):
            s_next = int(spec.next_state[s, i])
            if s_next == UNDEFINED:
                continue  # input not applicable to the specification
            s_out = int(spec.output[s, i])

            m_next, m_out = delta.get((m, i), (int(spec.next_state[m, i]),
                                               int(spec.output[m, i])))
            if m_next == UNDEFINED or m_out != s_out:
                return True

            pair = (s_next, m_next)
            if pair not in seen:
                seen.add(pair)
                queue.append(pair)
    return False


def _other(rng: random.Random, current: int, n: int, lo: int = 0):
    """A value in ``[lo, lo+n)`` different from ``current``; None if impossible."""
    if n < 2:
        return None
    v = rng.randrange(lo, lo + n)
    while v == current:
        v = rng.randrange(lo, lo + n)
    return v


def generate(spec: Spec, transitions: list, *, count: int, fault_type: int = 3,
             faults_per_mutant: int = 3, seed: int = 42,
             max_attempts_factor: int = 100):
    """Yield ``count`` mutants of ``spec`` as lists of transition rows.

    ``transitions`` is the original list of ``[src, dst, input, output]`` token
    rows, so emitted mutants keep the specification's line ordering and label
    alphabet.
    """
    rng = random.Random(seed)
    s_index = {lab: k for k, lab in enumerate(spec.state_labels)}
    i_index = {lab: k for k, lab in enumerate(spec.input_labels)}
    n_trans = len(transitions)

    cap = n_trans * (2 if fault_type == 2 else 1)
    n_faults = min(faults_per_mutant, cap)
    if n_faults < 1:
        return

    seen: set = set()
    produced = 0
    attempts = 0
    limit = count * max_attempts_factor

    while produced < count and attempts < limit:
        attempts += 1
        rows = [list(r) for r in transitions]
        k = rng.randint(1, n_faults)
        chosen = rng.sample(range(n_trans), min(k, n_trans))

        for idx in chosen:
            row = rows[idx]
            do_out = fault_type in (0, 3) or (fault_type == 2 and rng.random() < 0.5)
            do_state = fault_type in (1, 3) or (fault_type == 2 and rng.random() < 0.5)
            if fault_type == 2 and not (do_out or do_state):
                do_out = True
            if do_out:
                v = _other(rng, int(row[3]), spec.n_outputs)
                if v is not None:
                    row[3] = str(v)
            if do_state:
                cur = s_index[int(row[1])]
                v = _other(rng, cur, spec.n_states)
                if v is not None:
                    row[1] = str(spec.state_labels[v])

        signature = tuple(
            (i, rows[i][1], rows[i][3])
            for i in range(n_trans)
            if rows[i][1] != transitions[i][1] or rows[i][3] != transitions[i][3]
        )
        if not signature or signature in seen:
            continue

        delta = {}
        for i, dst, out in signature:
            delta[(s_index[int(rows[i][0])], i_index[rows[i][2]])] = (
                s_index[int(dst)], int(out))
        if not is_distinguishable(spec, delta):
            continue  # equivalent mutant

        seen.add(signature)
        produced += 1
        yield rows


def generate_file(spec_path, out_path, *, count: int, fault_type: int = 3,
                  faults_per_mutant: int = 3, seed: int = 42,
                  batch: int = 1000) -> int:
    """Generate ``count`` mutants per specification in ``spec_path``.

    Mutants are streamed to ``out_path`` in batches so that memory stays flat
    regardless of how many are requested.
    """
    from .io import read_specs

    specs = {s.fsm_id: s for s in read_specs(spec_path)}
    blocks = list(_iter_blocks(spec_path))
    total = 0

    with open(out_path, "w") as fh:
        for header, rows in blocks:
            fsm_id = int(header[0])
            spec = specs[fsm_id]
            pending = []
            for mutant_rows in generate(spec, rows, count=count,
                                        fault_type=fault_type,
                                        faults_per_mutant=faults_per_mutant,
                                        seed=seed + fsm_id):
                pending.append(mutant_rows)
                if len(pending) >= batch:
                    for m in pending:
                        write_fsm(fh, header, m)
                    total += len(pending)
                    pending.clear()
            for m in pending:
                write_fsm(fh, header, m)
            total += len(pending)
    return total
