"""Greedy look-ahead search.

Each round, every input defined at the current state is evaluated for the
number of mutants it kills immediately. The best is taken. Ties are broken by
looking one step further: the tied input whose successor state offers the most
kills on its own best next input wins; if still tied, the lowest-indexed input
is taken so the algorithm is deterministic.

There is no depth cap -- the sequence grows one input per round until the time
budget is spent or every mutant is dead.

Dead ends
---------
On a partial machine the walk can reach a state with no defined inputs. The
search then restarts from the initial state, keeping already-killed mutants
dead. That makes the output a *suite* of sequences rather than one sequence, so
each completed segment is replayed against the full mutant population to score
it fairly as a standalone test. Set ``restart=False`` to stop at the first dead
end instead.
"""

from __future__ import annotations

import numpy as np

from ..core import (KillCondition, Survivors, Tracker, initial_survivors,
                    kills_of_sequence, step)
from ..io import MutantSet, Spec


def _best_immediate(spec, mutants, state, surv, kill) -> int:
    """Most mutants killable from ``state`` by a single input."""
    best = 0
    for inp in spec.defined_inputs(state):
        _, _, killed = step(spec, mutants, state, surv, int(inp), kill)
        if killed.size > best:
            best = int(killed.size)
    return best


def run_lookahead(spec: Spec, mutants: MutantSet, *, budget: float,
                  kill: KillCondition, max_depth: int | None = None,
                  restart: bool = True, **_) -> tuple:
    tracker = Tracker(spec, mutants, "Lookahead-Greedy", budget, kill)
    total = mutants.n_mutants

    state = spec.initial
    surv = initial_survivors(mutants, spec)
    segment: tuple = ()
    reason = "no progress possible"

    def close_segment():
        """Score the finished segment as a standalone distinguishing test."""
        if segment:
            tracker.observe_path(segment, kills_of_sequence(spec, mutants,
                                                            segment, kill))

    while True:
        tracker.checkpoint()
        if tracker.expired():
            reason = "time budget reached"
            break
        if surv.count == 0:
            reason = "all mutants killed"
            break
        if max_depth is not None and len(segment) >= max_depth:
            reason = "max depth reached"
            break

        actions = spec.defined_inputs(state)
        if actions.size == 0:
            close_segment()
            if not restart:
                reason = "dead end reached"
                break
            state = spec.initial
            surv = Survivors(idx=surv.idx,
                             state=np.full(surv.count, spec.initial, np.int32))
            segment = ()
            continue

        # -- depth 1: immediate kills for every input ----------------------
        candidates = []
        for inp in actions:
            inp = int(inp)
            nxt, nsurv, killed = step(spec, mutants, state, surv, inp, kill)
            tracker.note_node(len(segment) + 1)
            candidates.append((int(killed.size), inp, nxt, nsurv, killed))

        best_kills = max(c[0] for c in candidates)
        tied = [c for c in candidates if c[0] == best_kills]

        # -- depth 2: tie-break on what the successor state offers ---------
        if len(tied) == 1:
            chosen = tied[0]
        else:
            chosen, best_future = tied[0], -1
            for c in tied:
                future = _best_immediate(spec, mutants, c[2], c[3], kill)
                tracker.note_node(len(segment) + 2)
                if future > best_future:
                    best_future, chosen = future, c

        _, inp, nxt, nsurv, killed = chosen
        segment = segment + (inp,)
        tracker.observe_step(killed, len(segment))
        state, surv = nxt, nsurv
        tracker.observe_path(segment, total - nsurv.count)

        if tracker.result.best_kills == total:
            reason = "all mutants killed by a single test"
            break

    close_segment()
    return tracker.finish(reason), tracker
