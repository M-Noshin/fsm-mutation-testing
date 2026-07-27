# Distinguishing FSM Mutants: Reinforcement Learning, Look-ahead, BFS and DFS

[![tests](https://github.com/M-Noshin/fsm-mutation-testing/actions/workflows/tests.yml/badge.svg)](https://github.com/M-Noshin/fsm-mutation-testing/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the code for the paper *Reinforcement Learning and
Look-ahead Approaches for Distinguishing Finite State Machine Mutants* by Uraz,
Noshin, and Khaled El-Fakih (American University of Sharjah).

---

## 1. What this software does, in plain terms

A **finite state machine (FSM)** is a model of a system: it reads a sequence of
inputs and, for each input, moves between states and produces an output.

A **mutant** is a copy of a correct FSM (the *specification*) with a small
deliberate fault introduced — for example, one transition sends the machine to
the wrong state, or prints the wrong output.

The central task of this software is:

> Given a correct specification and a large collection of mutants, find a short
> input sequence whose output reveals the fault — that is, a sequence that
> produces one output on the specification and a *different* output on the
> mutant. Such a sequence is called a **distinguishing test**, and a mutant it
> exposes is said to be **killed**.

The software implements **four different strategies** for searching for such
sequences, runs them under identical conditions, and records how well each one
performs. The four strategies are:

| Strategy | Command name | What it does |
|---|---|---|
| Breadth-first search | `bfs` | Explores all short sequences before longer ones. |
| Depth-first search | `dfs` | Explores one branch as deep as allowed before backtracking. |
| Q-learning (reinforcement learning) | `rl` | Learns, over repeated attempts, which inputs tend to expose faults. |
| Greedy look-ahead | `la` | At each step, picks the input that kills the most remaining mutants. |

None of the four is guaranteed to find the best possible sequence — the
underlying problems are known to be computationally hard (PSPACE-complete and
NP-complete respectively). They are therefore *heuristics*, and the purpose of
this software is to measure how they trade off four quantities against one
another: **how many mutants they kill, how long their tests are, how much time
they take, and how much memory they use.**

**A note on trust.** Every result this software reports can be independently
re-checked. Each row of output stores the exact input sequence that was found,
and a separate verification tool (`fsm-validate`) replays that sequence with a
completely independent interpreter to confirm the reported score is correct.
This verification also runs automatically in continuous integration.

---

## 2. Table of contents

1. [What this software does](#1-what-this-software-does-in-plain-terms)
2. Table of contents *(you are here)*
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Quick start: your first result in three steps](#5-quick-start-your-first-result-in-three-steps)
6. [Preparing the input files](#6-preparing-the-input-files)
7. [Running experiments: all options](#7-running-experiments-all-options)
8. [Reading the results](#8-reading-the-results)
9. [Building a summary report](#9-building-a-summary-report)
10. [Verifying results independently](#10-verifying-results-independently)
11. [Methodology: four decisions that affect the numbers](#11-methodology-four-decisions-that-affect-the-numbers)
12. [Repository layout](#12-repository-layout)
13. [Running the tests](#13-running-the-tests)
14. [Reproducibility](#14-reproducibility)
15. [Citation](#15-citation)

---

## 3. Requirements

- **Python 3.10 or newer.**
- Four packages, installed automatically in the next step: `numpy`, `pandas`,
  `openpyxl`, `psutil`.

No other software is required.

---

## 4. Installation

Run these three commands in a terminal:

```bash
git clone https://github.com/M-Noshin/fsm-mutation-testing
cd fsm-mutation-testing
pip install -e .
```

The `pip install -e .` step installs the package and its dependencies, and
creates three convenience commands: `fsm-run`, `fsm-report`, and
`fsm-validate`. Throughout this README, each convenience command is shown
alongside its longer equivalent (for example, `fsm-run` is the same as
`python -m fsm_mutation.run`); use whichever you prefer.

---

## 5. Quick start: your first result in three steps

This section takes you from a fresh install to a verified result. It uses the
smallest example so it finishes quickly.

**Step 1 — Generate a set of mutants** to test against. The command below
creates 5,000 mutants of the complete-machine specification and saves them to a
file:

```bash
python -c "
from fsm_mutation.generate import generate_file
generate_file('data/fsms/128state_fsms_complete.txt',
              'data/mutants/complete_5000.txt',
              count=5000, seed=42)"
```

**Step 2 — Run the four search strategies** against those mutants:

```bash
fsm-run --machines complete --algorithm all --mutants 5000
```

While this runs, it prints progress to the screen and writes one row of results
per (machine, strategy) into a spreadsheet under the `results/` folder. Because
results are written row by row, an interrupted run keeps everything it had
already finished.

**Step 3 — Verify the results** by replaying every recorded sequence with an
independent checker:

```bash
fsm-validate results/complete_5000_output.xlsx
```

If the reported scores match the independent replay, the command reports
success. You now have a verified spreadsheet of results. The remaining sections
explain each of these steps in full.

---

## 6. Preparing the input files

There are two kinds of input: **specification files** (the correct machines)
and **mutant files** (the faulty copies you generate from them).

### 6a. Specification files

Specification files live in `data/fsms/`. The runner looks for three specific
file names, selected with the `--machines` option:

| `--machines` value | expected file in `data/fsms/` | search depth cap |
|---|---|---|
| `complete` | `128state_fsms_complete.txt` | ⌈n / 3⌉ |
| `partial`  | `128state_fsms_partial.txt`  | ⌈n / 3⌉ |
| `cerny`    | `cerny_machines.txt`         | 4 |
| `custom`   | any file you pass with `--spec-file` | value of `--depth` |

**File format.** A specification file holds one or more machines. Each machine
begins with a header line of six integers:

```
<fsm_id> <num_states> <num_transitions> <num_inputs> <num_outputs> <void>
```

followed by one line per transition, each with four tokens:

```
<source_state> <target_state> <input> <output>
```

Machines are separated by blank lines. Partial machines simply leave out the
transitions that are undefined. Input labels may be letters or digits; state
labels must be integers, and the initial state is the numerically smallest
state label. This format is also documented in
[`data/fsms/README.md`](data/fsms/README.md).

### 6b. Generating mutant files

Mutants are created from a specification with `generate_file`. Mutant files are
**not** stored in this repository; you regenerate them from a seed, which
guarantees you get exactly the same set every time (see
[Reproducibility](#14-reproducibility)).

```bash
python -c "
from fsm_mutation.generate import generate_file
generate_file('data/fsms/128state_fsms_complete.txt',
              'data/mutants/complete_50000.txt',
              count=50000, fault_type=3, faults_per_mutant=3, seed=42)"
```

The `fault_type` argument controls what kind of fault is injected:

| `fault_type` | Name | What changes |
|---|---|---|
| 0 | Output fault | A transition prints a different output. |
| 1 | Transfer fault | A transition moves to a different next state. |
| 2 | Mixed | One or two faults on the chosen transition. |
| 3 (default) | Transfer **and** output fault | Both, on the same transition. |

Every generated mutant is guaranteed to be **distinct** and **non-equivalent**
to the specification — meaning at least one input sequence can tell them apart.
Equivalence is decided by a breadth-first walk of the product automaton that
compares outputs.

---

## 7. Running experiments: all options

A single command runs an experiment. You choose a machine family, one or all
strategies, and a number of mutants:

```bash
# All four strategies, complete 128-state machines, 50,000 mutants
fsm-run --machines complete --algorithm all --mutants 50000

# Q-learning only, on the partial machines, 30,000 mutants
fsm-run --machines partial --algorithm rl --mutants 30000

# Cerny machines (the paper uses a depth cap of 4 for these)
fsm-run --machines cerny --algorithm all --mutants 10000
```

The full list of options:

| Option | Default | Meaning |
|---|---|---|
| `--machines` | `complete` | Machine family: `complete`, `partial`, `cerny`, or `custom`. |
| `--spec-file` | — | Specification file to use when `--machines custom`. |
| `--mutant-file` | — | Use an existing mutant file instead of generating one. |
| `--algorithm` | `all` | Strategy to run: `bfs`, `dfs`, `rl`, `la`, or `all`. |
| `--mutants` | `50000` | Number of mutants to generate per machine. |
| `--limit-mutants` | — | Cap the number of mutants actually used (useful for quick trials). |
| `--machine` | all | Run only the given machine id; repeat the flag for several. |
| `--depth` | — | Search depth cap (used with `--machines custom`). |
| `--budget` | `360` | Time budget per search, in seconds. |
| `--seed` | `42` | Random seed for mutant generation. |
| `--kill-condition` | `output` | When a mutant counts as killed (see [Section 11](#11-methodology-four-decisions-that-affect-the-numbers)). |
| `--node-storage` | `survivors` | `survivors` or `replay`; `replay` uses far less memory. |
| `--episodes` | `10000` | Number of learning episodes for Q-learning. |
| `--no-la-restart` | off | Disable look-ahead's restart behaviour. |
| `--no-isolate` | off | Run without subprocess isolation (faster, but memory numbers become meaningless). |
| `--data-dir` | `data` | Folder holding `fsms/` and `mutants/`. |
| `--out-dir` | `results` | Folder for the output spreadsheets. |

**One point worth knowing:** greedy look-ahead (`la`) always searches without a
depth cap. With a depth cap it usually stops before finding a solution, which
is the behaviour described in the paper.

Results are appended row by row to a spreadsheet named
`results/<class>_<mutants>_<kill-condition>.xlsx`.

---

## 8. Reading the results

Each run adds **one row per (machine, strategy)** to the output spreadsheet.
The columns you will care about most:

| Column | Meaning |
|---|---|
| `Final Mutation Score (%)` | Percentage of mutants killed by the **single best** distinguishing test found. |
| `Final DS Length` | Length of that single best test. (`DS` = distinguishing sequence.) |
| `Best Sequence` | The test itself. You can replay it to confirm the row. |
| `Suite Score (%)` | Percentage killed by the **union of all** sequences explored during the search. |
| `Suite Size (# tests)` | How many tests that union keeps. |
| `Suite Cost (# inputs)` | Total number of inputs across those tests. |
| `Peak Memory (MB)` | Highest memory use of the isolated run, in megabytes. |
| `Terminated At (sec)` | When the search stopped. |
| `Stop Reason` | Why it stopped. |
| `... after 30s / 1min / …` | The same measurements sampled at nine checkpoints; left blank after the run ends. |

Additional columns — `Machine Class`, `Kill Condition`, `Seed`,
`Max search depth`, and `Mutant File` — record the exact configuration, so rows
from different runs can safely be combined into one table.

---

## 9. Building a summary report

To combine one or more result spreadsheets into a single report with charts:

```bash
fsm-report results/*.xlsx -o results/report.xlsx
```

The report contains one sheet per machine family. Each sheet shows average
mutation score against time, average score and test length per machine, and
average memory against time, together with charts.

---

## 10. Verifying results independently

This is the check that makes the results trustworthy. It replays the exact
sequence recorded in each row and confirms the score:

```bash
fsm-validate results/complete_50000_output.xlsx
```

The validator reads `Best Sequence` from every row, replays it against the
mutant file using a plain dictionary interpreter that **shares no code** with
the search strategies, and confirms that the mutation score matches what was
reported. If any row does not match, the command fails and tells you which one.

---

## 11. Methodology: four decisions that affect the numbers

The four strategies differ **only** in how they explore. Everything else — the
initial state, the mutant set, the rule for killing, the clock — is shared code
in `core.py` and is identical for all four.

Four modelling decisions are stated explicitly here, because earlier versions
of this code handled them differently. Numbers produced under the old choices
are **not** directly comparable to numbers produced now. Full detail is in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

### Decision 1 — A mutant is killed only when the *output* differs

> A mutant `M` is killed by a sequence `s` if, and only if, applying `s` from
> the initial state produces different **output sequences** on the
> specification and on `M`.

A transfer fault (wrong next state) produces nothing visible at the step where
it is taken. It is detected only later, if the machine's divergence eventually
causes a different output. Counting a wrong next state as an immediate kill
would credit a test with faults it cannot actually observe, which is why this
software does not do so.

The old behaviour is still available with `--kill-condition output_or_state`.
It changes nothing for fault types 0 and 3 (where the output always changes as
well), but in our measurements it inflates fault type 1 by roughly 18% and
fault type 2 by roughly 7%.

### Decision 2 — The single best test and the whole suite are reported separately

Tree searches (BFS, DFS) naturally produce a **suite** — the union of every
mutant killed anywhere in the search tree. Q-learning and look-ahead produce a
**single** sequence. These are two different quantities, and comparing one
against the other would be unfair, so every run reports both:

| Columns | What they measure |
|---|---|
| `Final Mutation Score (%)`, `Final DS Length` | The single best distinguishing test. |
| `Suite Score (%)`, `Suite Size`, `Suite Cost` | The union over all sequences explored. |

The difference can be large. On an 8-state machine with 2,000 mutants and a
depth cap of 6, the BFS *suite* kills 100% of mutants while its *single best*
test kills only 51%.

### Decision 3 — `DS Length` is exactly the length of the reported test

`Final DS Length` is simply the number of inputs in `Best Sequence`, and
nothing else. Because `Best Sequence` is written into the spreadsheet, any row
can be re-scored independently. (In an earlier version, this column tracked the
depth at which an internal counter last increased, which for BFS is just the
frontier depth — a different and misleading quantity.)

### Decision 4 — Memory is measured in an isolated process

CPython does not always return freed memory to the operating system, so two
searches run one after another in the same interpreter can contaminate each
other's memory readings. In testing, a search that used 33 MB on its own
reported 2,666 MB when it ran second in the same process. To avoid this, each
(machine, strategy) pair runs in its **own subprocess**, and `Peak Memory (MB)`
comes from the operating system's own accounting (`getrusage`).

The `--no-isolate` option turns this off. It is faster, but the memory columns
then become meaningless. Checkpoints recorded after a run ends are left blank
rather than filled with a stale reading; `Terminated At (sec)` and `Stop
Reason` explain what happened.

### A note on how mutants are stored

Each mutant is stored as a small **difference** from the specification — only
the transitions that actually changed — and the search keeps track of only the
mutants still alive, so its memory footprint shrinks as the search proceeds.
For 50,000 mutants of a 128-state machine this is a few megabytes rather than
several gigabytes. As a result, the memory column reflects the *strategy*, not
the storage scheme.

The `--node-storage replay` option removes the per-node survivor lists entirely
and recomputes them on demand. It produces identical results (there is a test
that confirms this) at much lower memory.

---

## 12. Repository layout

```
src/fsm_mutation/
  io.py             Parsing; the Spec and MutantSet data structures
  core.py           Kill rules, metrics, result rows, spreadsheet output
  generate.py       Mutant generation and equivalence checking
  algorithms/
    tree.py         BFS and DFS (shared traversal, different frontier order)
    qlearning.py    Q-learning
    lookahead.py    Greedy look-ahead with one-step tie-breaking
  run.py            Command-line entry point
  report.py         Aggregated report with charts
  validate.py       Independent replay verification
  _worker.py        Runs a single search in its own subprocess
tests/              Hand-checked semantics and cross-strategy invariants
docs/METHODOLOGY.md What every column means and why
data/fsms/          Specification files
data/mutants/       Generated mutants (not committed; regenerate from a seed)
results/            Output spreadsheets
```

---

## 13. Running the tests

```bash
pip install -e ".[dev]"
pytest
```

---

## 14. Reproducibility

Every run records its seed, kill condition, depth cap, mutant file, and the
sequence it found. Mutant generation is seeded per machine, so
`generate_file(..., seed=42)` recreates a mutant set exactly. This is why mutant
files are not committed to the repository — the seed alone reproduces them.

---

## 15. Citation

```bibtex
@article{uraz_noshin_elfakih_fsm_mutants,
  title  = {Reinforcement Learning and Look-ahead Approaches for
            Distinguishing Finite State Machine Mutants},
  author = {Uraz and Noshin and El-Fakih, Khaled},
  note   = {American University of Sharjah}
}
```
