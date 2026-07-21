"""Q-learning search for a distinguishing test.

A state of the MDP is the pair ``(s, M_r)`` from the paper: the specification's
current state together with the set of not-yet-killed mutants and the state
each of them occupies. An action is an input defined at ``s``.

Reward (paper, section 3.1), with ``C = 2``::

    R = -100                                  if the action is undefined
    R = -10                                   if no mutant is killed
    R = (|M_r| - |M_r'|) / |M_r| * C          otherwise

Each episode is a walk from the initial state, so what the agent learns is
scored against a real input sequence rather than against the union of
everything the search happened to touch.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

from ..core import (KillCondition, Tracker, initial_survivors, step)
from ..io import MutantSet, Spec


def _node_key(state: int, depth: int, surv) -> tuple:
    """Identity of a search node.

    The surviving-mutant configuration is summarised by a 128-bit digest;
    holding the raw arrays as dictionary keys would cost more memory than the
    search itself.
    """
    h = hashlib.blake2b(surv.idx.tobytes(), digest_size=8).digest()
    h += hashlib.blake2b(surv.state.tobytes(), digest_size=8).digest()
    return (state, depth, surv.count, h)


def run_qlearning(spec: Spec, mutants: MutantSet, *, budget: float,
                  max_depth: int | None, kill: KillCondition,
                  seed: int = 42, episodes: int = 10_000,
                  alpha: float = 0.7, gamma: float = 0.7,
                  epsilon: float = 1.0, epsilon_decay: float = 0.999,
                  min_epsilon: float = 0.01, C: float = 2.0, **_) -> tuple:
    rng = random.Random(seed)
    tracker = Tracker(spec, mutants, "Q-Learning", budget, kill)
    total = mutants.n_mutants

    # Q[node_key][input] -- created lazily, one entry per visited node.
    q: dict = {}
    root_surv = initial_survivors(mutants, spec)
    reason = "episode budget exhausted"
    episode = 0

    while episode < episodes:
        episode += 1
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        state = spec.initial
        surv = root_surv
        path: tuple = ()

        while True:
            tracker.checkpoint()
            if tracker.expired():
                reason = "time budget reached"
                break
            if max_depth is not None and len(path) >= max_depth:
                break
            if surv.count == 0:
                break

            actions = spec.defined_inputs(state)
            if actions.size == 0:
                break

            key = _node_key(state, len(path), surv)
            qs = q.get(key)
            if qs is None:
                qs = q[key] = {int(a): 0.0 for a in actions}

            if rng.random() < epsilon:
                action = rng.choice(list(qs))
            else:
                action = max(qs, key=qs.__getitem__)

            before = surv.count
            nxt, nsurv, killed = step(spec, mutants, state, surv, action, kill)
            npath = path + (action,)

            tracker.note_node(len(npath))
            tracker.observe_step(killed, len(npath))
            tracker.observe_path(npath, total - nsurv.count)

            # Reward
            if killed.size == 0:
                reward = -10.0
            else:
                reward = (killed.size / before) * C

            # Bootstrap from the successor's best known value.
            nkey = _node_key(nxt, len(npath), nsurv)
            nqs = q.get(nkey)
            future = max(nqs.values()) if nqs else 0.0

            qs[action] += alpha * (reward + gamma * future - qs[action])

            state, surv, path = nxt, nsurv, npath

            if tracker.result.best_kills == total:
                reason = "all mutants killed by a single test"
                break

        if tracker.expired():
            reason = "time budget reached"
            break
        if tracker.result.best_kills == total:
            reason = "all mutants killed by a single test"
            break

    result = tracker.finish(reason)
    result.episodes = episode
    return result, tracker
