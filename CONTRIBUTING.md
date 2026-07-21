# Contributing

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## The one rule

All four algorithms must share the semantics in `src/fsm_mutation/core.py`.
If BFS, DFS, Q-learning and look-ahead do not step the mutant population
through the same `step()` function, their scores stop being comparable and the
central claim of the paper stops being testable.

Adding an algorithm means adding a module under `algorithms/` that calls
`core.step` and reports through `core.Tracker`. It does not mean writing a new
kill-detection routine.

## Before opening a pull request

1. `pytest` passes.
2. If you touched anything in `core.py`, `io.py`, or an algorithm, run the
   independent replay check:

   ```bash
   python -m fsm_mutation.run --machines custom --spec-file data/fsms/<f>.txt \
       --mutant-file data/mutants/<m>.txt --algorithm all --mutants 2000 --budget 20
   python -m fsm_mutation.validate results/<out>.xlsx --data-dir data
   ```

   Every row must report `ok`. This catches scoring bugs that unit tests miss,
   because the validator re-derives the score with a separate implementation.
3. If you changed what a column means, update `docs/METHODOLOGY.md`. Results
   produced under different measurement decisions are not comparable, and the
   only defence is writing the decisions down.

## Adding a machine family

Add an entry to `MACHINE_CLASSES` in `run.py` with the specification filename
and the depth rule, and drop the file into `data/fsms/`.

## Style

Docstrings explain *why* a decision was made where the reason is not obvious
from the code. Several choices here look arbitrary until you know what they
are guarding against — subprocess isolation, sparse deltas, blank
post-termination intervals — so those carry an explanation.
