"""Reinforcement learning and look-ahead approaches for distinguishing FSM mutants.

Companion code for Uraz, Noshin & El-Fakih.

Quick start::

    from fsm_mutation.io import read_specs, read_mutants
    from fsm_mutation.core import KillCondition
    from fsm_mutation.algorithms import ALGORITHMS

    spec = read_specs("data/fsms/128state_complete.txt")[0]
    mutants = read_mutants("data/mutants/complete_50000.txt", spec)
    result, tracker = ALGORITHMS["rl"](
        spec, mutants, budget=360, max_depth=43,
        kill=KillCondition.OUTPUT, seed=42,
    )
    print(result.best_kills, result.ds_length, spec.render(result.best_sequence))
"""

__version__ = "1.0.0"
