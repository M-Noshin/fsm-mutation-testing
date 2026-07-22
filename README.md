# Distinguishing FSM Mutants — RL, Look-ahead, BFS and DFS

[![tests](https://github.com/OWNER/fsm-mutation-testing/actions/workflows/tests.yml/badge.svg)](https://github.com/OWNER/fsm-mutation-testing/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code for *Reinforcement learning and look-ahead approaches for
distinguishing finite state machine mutants* (Uraz, Noshin & Khaled El-Fakih,
American University of Sharjah).

Given a deterministic FSM specification `S` and a collection of mutants
`M₁ … Mₖ`, the goal is a short input sequence — a *distinguishing test* — whose
output differs between `S` and as many mutants as possible. Four search
strategies are implemented and compared: breadth-first search, depth-first
search, Q-learning, and greedy look-ahead.

The bounded sequence problem is PSPACE-complete and the bounded mutant-killing
problem is NP-complete, so all four are heuristics; the question the code
answers is how they trade mutation score, test length, time and memory against
one another.

**Every reported result is replayable.** Each row of output carries the actual
input sequence found, and `fsm_mutation.validate` re-scores it with an
independent interpreter that shares no code with the search. That check runs in
CI.

---

## Install

```bash
git clone https://github.com/<you>/fsm-mutation-testing
cd fsm-mutation-testing
pip install -e .
```

Python 3.10+. Dependencies: numpy, pandas, openpyxl, psutil.

## Run an experiment

Everything is driven by one command. Pick a machine family, an algorithm, and
a mutant count:

```bash
# every algorithm on the complete 128-state machines, 50 000 mutants
python -m fsm_mutation.run --machines complete --algorithm all --mutants 50000

# Q-learning only, on the partial machines
python -m fsm_mutation.run --machines partial --algorithm rl --mutants 30000

# Cerny machines (the paper uses a depth cap of 4 for these)
python -m fsm_mutation.run --machines cerny --algorithm all --mutants 10000
```

| `--machines` | specification file expected under `data/fsms/` | depth cap |
|---|---|---|
| `complete` | `128state_fsms_complete.txt` | `⌈n/3⌉` |
| `partial`  | `128state_fsms_partial.txt`  | `⌈n/3⌉` |
| `cerny`    | `cerny_machines.txt`         | `4` |
| `custom`   | whatever you pass to `--spec-file` | `--depth` |

`--algorithm` takes `bfs`, `dfs`, `rl`, `la`, or `all`. Look-ahead always runs
uncapped: with a depth cap it usually terminates before finding a solution,
which is the behaviour described in the paper.

Results are appended row-by-row to
`results/<class>_<mutants>_<kill-condition>.xlsx`, so a run that is interrupted
keeps everything it had already finished.

### Generating mutants

```bash
python -c "
from fsm_mutation.generate import generate_file
generate_file('data/fsms/128state_fsms_complete.txt',
              'data/mutants/complete_50000.txt',
              count=50000, fault_type=3, faults_per_mutant=3, seed=42)"
```

| fault type | meaning |
|---|---|
| 0 | output fault — change a transition's output |
| 1 | transfer fault — change a transition's next state |
| 2 | mixed — one or two faults per chosen transition |
| 3 | transfer **and** output fault on the same transition |

Every mutant is guaranteed distinct and non-equivalent; equivalence is decided
by a breadth-first walk of the product automaton comparing **outputs**.

### Building the report

```bash
python -m fsm_mutation.report results/*.xlsx -o results/report.xlsx
```

One sheet per machine class, each with average mutation score vs. time, average
score and DS length per machine, and average memory vs. time, with charts.

### Verifying results

```bash
python -m fsm_mutation.validate results/complete_50000_output.xlsx
```

Every row records the actual input sequence found. The validator replays it
against the mutant file using a plain dictionary interpreter that shares no
code with the search, and checks the mutation score matches.

---

## Output columns

Each run appends one row per (machine, algorithm). The columns that matter:

| column | meaning |
|---|---|
| `Final Mutation Score (%)` | % of mutants killed by the **best single** distinguishing test |
| `Final DS Length` | length of that test |
| `Best Sequence` | the test itself — replay it to verify the row |
| `Suite Score (%)` | % killed by the **union** of every sequence explored |
| `Suite Size (# tests)` / `Suite Cost (# inputs)` | how many tests the suite keeps and what it costs |
| `Peak Memory (MB)` | high-water mark of the isolated run process |
| `Terminated At (sec)` / `Stop Reason` | when and why the search stopped |
| `... after 30s / 1min / …` | the same metrics sampled at nine checkpoints, blank once the run has ended |

`Machine Class`, `Kill Condition`, `Seed`, `Max search depth` and `Mutant File`
record the configuration, so rows from different runs can be pooled safely.

## What is measured, and why it changed

Full detail in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Summary:

The four algorithms differ only in how they explore. Everything else — the
initial state, the mutant set, the kill condition, the clock — comes from
`core.py` and is identical across them. Four decisions are worth stating
explicitly, because earlier versions of this code handled them differently and
the resulting numbers are not comparable to these.

### 1. A mutant is killed when the **output** differs

> `M` is killed by `s` ⟺ applying `s` from the initial state to `S` and to `M`
> yields different output sequences.

A transfer fault produces nothing observable at the step where it is
traversed; it is detected only later, if the divergent state eventually emits a
different output. Counting a next-state difference as a kill therefore credits
the test with faults it cannot detect.

`--kill-condition output_or_state` restores the older behaviour. It makes no
difference for fault types 0 and 3 (where the output always changes too) but
inflates fault type 1 by roughly 18% and fault type 2 by 7% in our measurements.

### 2. Single test and suite are reported separately

Tree searches naturally produce a *suite* — the union of everything killed
anywhere in the tree — while Q-learning and look-ahead produce a *single*
sequence. These are different quantities and scoring one against the other is
not a fair comparison, so every run reports both:

| column | meaning |
|---|---|
| `Final Mutation Score (%)`, `Final DS Length` | best single distinguishing test |
| `Suite Score (%)`, `Suite Size`, `Suite Cost` | union over all sequences explored, number of tests kept, total inputs |

The gap is large. On a 8-state machine with 2 000 mutants and a depth cap of 6,
BFS's suite kills 100% while its best single test kills 51%.

### 3. `DS Length` is the length of the reported test

It is `len(best_sequence)` and nothing else, and `Best Sequence` is written to
the workbook so any row can be independently re-scored. Previously this column
tracked the depth at which a cumulative counter last increased, which in BFS is
just the frontier depth.

### 4. Memory is measured in an isolated process

CPython does not return freed arenas to the operating system, so consecutive
runs in one interpreter inherit each other's resident size. In testing, a
search measuring 33 MB alone reported 2 666 MB when it ran second in the same
process. Each (machine, algorithm) pair therefore runs in its own subprocess
and reports `Peak Memory (MB)` from `getrusage`. `--no-isolate` is faster but
makes the memory columns meaningless.

Intervals after a run terminates are left blank rather than back-filled with a
stale reading; `Terminated At (sec)` and `Stop Reason` say what happened.

### Representation

Mutants are stored as a sparse delta against the specification — only the
transitions that actually differ — and the search carries only the *surviving*
mutants, so its footprint shrinks as the search progresses. For 50 000 mutants
of a 128-state machine this is a few megabytes rather than several gigabytes,
which means the memory column reflects the algorithm rather than the storage
scheme.

`--node-storage replay` drops the per-node survivor arrays entirely and
recomputes them from the path on expansion. It gives identical results (there
is a test for this) at much lower memory.

---

## Repository layout

```
src/fsm_mutation/
  io.py             parsing, Spec and MutantSet
  core.py           kill semantics, metrics, result rows, Excel output
  generate.py       mutant generation and equivalence checking
  algorithms/
    tree.py         BFS and DFS (shared traversal, different frontier order)
    qlearning.py    Q-learning
    lookahead.py    greedy look-ahead with one-step tie-breaking
  run.py            CLI
  report.py         aggregated workbook with charts
  validate.py       independent replay verification
  _worker.py        single-run subprocess
tests/              hand-checked semantics + cross-algorithm invariants
docs/METHODOLOGY.md what every column means and why
data/fsms/          specifications
data/mutants/       generated mutants (not committed; regenerate with a seed)
results/            output workbooks
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Reproducibility

Every run records its seed, kill condition, depth cap, mutant file and the
sequence found. Mutant generation is seeded per machine, so
`generate_file(..., seed=42)` reproduces a mutant set exactly without needing
the file itself to be committed.

## Citation

```bibtex
@article{uraz_noshin_elfakih_fsm_mutants,
  title  = {Reinforcement learning and look-ahead approaches for
            distinguishing finite state machine mutants},
  author = {Uraz and Noshin and El-Fakih, Khaled},
  note   = {American University of Sharjah}
}
```
