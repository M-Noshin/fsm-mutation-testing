"""Breadth-first and depth-first tree search.

Both explore the same tree of input sequences and differ only in the order in
which the frontier is drained, so they share one implementation. This makes the
BFS/DFS comparison a comparison of traversal order and nothing else.

Node storage
------------
``node_storage="survivors"`` keeps the surviving-mutant array on every frontier
node. This is the faithful reading of the algorithm -- the search really does
have to remember where each mutant is -- and it is what the memory figures
measure.

``node_storage="replay"`` keeps only the input path and recomputes the
survivors by replaying it from the initial state when the node is expanded.
Paths are short, so the replay is cheap, and frontier memory becomes
negligible. Use it when memory, not traversal order, is the binding constraint.
"""

from __future__ import annotations

from collections import deque

from ..core import (KillCondition, Survivors, Tracker, initial_survivors, step)
from ..io import Spec, MutantSet


def _search(spec: Spec, mutants: MutantSet, *, order: str, budget: float,
            max_depth: int | None, kill: KillCondition,
            node_storage: str = "survivors", **_) -> tuple:
    algorithm = "BFS" if order == "bfs" else "DFS"
    tracker = Tracker(spec, mutants, algorithm, budget, kill)
    total = mutants.n_mutants

    root_surv = initial_survivors(mutants, spec)
    keep = node_storage == "survivors"

    # frontier entries: (spec_state, path, survivors_or_None)
    frontier = deque([(spec.initial, (), root_surv if keep else None)])
    pop = frontier.popleft if order == "bfs" else frontier.pop

    tracker.note_node(0)
    tracker.observe_path((), 0)
    reason = "frontier exhausted"

    while frontier:
        tracker.checkpoint()
        if tracker.expired():
            reason = "time budget reached"
            break

        state, path, surv = pop()
        depth = len(path)
        if max_depth is not None and depth >= max_depth:
            continue

        if surv is None:  # replay mode
            surv = root_surv
            s = spec.initial
            for inp in path:
                s, surv, _ = step(spec, mutants, s, surv, inp, kill)

        if surv.count == 0:
            # Nothing left to distinguish along this branch.
            continue

        for inp in spec.defined_inputs(state):
            inp = int(inp)
            nxt, nsurv, killed = step(spec, mutants, state, surv, inp, kill)
            npath = path + (inp,)

            tracker.note_node(depth + 1)
            tracker.observe_step(killed, len(npath))
            tracker.observe_path(npath, total - nsurv.count)

            frontier.append((nxt, npath, nsurv if keep else None))

            if tracker.result.best_kills == total:
                reason = "all mutants killed by a single test"
                frontier.clear()
                break

        if tracker.expired():
            reason = "time budget reached"
            break

    return tracker.finish(reason), tracker


def run_bfs(spec, mutants, **kw):
    return _search(spec, mutants, order="bfs", **kw)


def run_dfs(spec, mutants, **kw):
    return _search(spec, mutants, order="dfs", **kw)
