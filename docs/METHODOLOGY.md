# Methodology and measurement decisions

This note records the decisions that determine what the numbers in
`results/` mean. It exists because four of them were made differently in the
earlier prototype scripts, and results produced under the two sets of
decisions are not comparable.

---

## 1. What counts as killing a mutant

> A mutant `M` is killed by an input sequence `σ` if and only if applying `σ`
> from the initial state to the specification `S` and to `M` yields **different
> output sequences**.

This matches the paper's problem statement and the criterion used by the mutant
generator's equivalence check.

**What changed.** The prototype declared a mutant killed when either the output
*or* the next state differed:

```python
if mutant_next_state != original_next_state or mutant_output != original_output:
    mutants_detected.add(idx)
```

A transfer fault produces nothing observable at the step where it is traversed.
The tester sees the same output from both machines; only later, if the
divergent state emits a different output, does the fault become detectable — and
it may never do so. Counting the state difference credits the test with faults
it cannot detect.

**Measured effect**, 8-state machine, 2 000 mutants, BFS at depth 6:

| generator fault type | correct | legacy | inflation |
|---|---:|---:|---:|
| 0 — output only | 55.70% | 55.70% | none |
| 1 — transfer only | 45.30% | 53.25% | ×1.18 |
| 2 — mixed | 48.00% | 51.50% | ×1.07 |
| 3 — transfer + output, same transition | 51.90% | 51.90% | none |

Fault types 0 and 3 are unaffected: the output always changes as well, so the
two conditions coincide. **Experiments run with `fault_type=3` are unaffected
by this correction.** Fault types 1 and 2 are affected.

`--kill-condition output_or_state` restores the legacy behaviour.

---

## 2. A single test and a test suite are different things

Tree searches explore many sequences and can report the union of everything
killed anywhere in the tree. Q-learning and look-ahead follow one walk and
report what that walk killed. These are different quantities.

**What changed.** In the prototype, BFS and DFS shared one `killed_mutants` set
across the whole search:

```python
killed_mutants = set()      # created once, outside the traversal
```

A mutant killed on branch `aab` stayed marked as killed while branch `bba` was
explored, so the reported score was the union over the tree while the reported
`DS Length` described a single path. The two describe different artifacts.

**Why the gap is large.** A sequence of length `l` traverses at most `l`
transitions. On a 128-state machine with ~600 transitions and mutants carrying
1–3 faults, the fraction of mutants with a fault on those `l` transitions is
approximately

```
1 − (1 − l/600)²  ≈  2.3%   for l = 7
```

So a genuine 7-input test kills on the order of 2% of a 50 000-mutant set. A
reported 100% at `DS Length` 7 is the union over the whole tree, not the score
of one test.

**What is reported now.** Both, for every algorithm:

| column | meaning |
|---|---|
| `Final Mutation Score (%)` | mutants killed by the best single sequence |
| `Final DS Length` | length of that sequence |
| `Best Sequence` | the sequence itself, so the row can be re-scored |
| `Suite Score (%)` | union over every sequence explored |
| `Suite Size (# tests)` | how many sequences the suite keeps |
| `Suite Cost (# inputs)` | total inputs across the suite |

The suite is built greedily: a sequence joins it only when it kills something
nothing before it killed. That makes it a real, replayable test suite with a
real cost rather than an abstract union.

Observed on an 8-state machine, 2 000 mutants, depth cap 7:

| algorithm | best single test | suite |
|---|---:|---|
| BFS | 57.25% (len 7) | 100% (21 tests, 51 inputs) |
| DFS | 57.25% (len 7) | 100% (21 tests, 129 inputs) |
| Q-learning | 57.25% (len 7) | 100% (21 tests, 78 inputs) |
| Look-ahead | 94.40% (len 19) | 94.40% (17 tests, 158 inputs) |

Note that look-ahead, which is uncapped, finds a far better *single* test — a
comparison that the earlier scoring could not express.

---

## 3. DS Length

`DS Length = len(best_sequence)`.

**What changed.** The prototype set

```python
depth_at_which_max_killed_so_far = current_depth + 1
```

whenever the cumulative counter increased. In BFS at least one new mutant dies
at essentially every level, so this tracked the frontier depth. Across 900
machine-interval cells of an earlier BFS run, `DS Length − Max Depth Reached`
was 1 in 817 cases and 0 in the remaining 83 — it was the frontier depth plus
one, by construction, and described no sequence.

---

## 4. Memory

Each `(machine, algorithm)` pair runs in its own subprocess and reports
`Peak Memory (MB)` from `getrusage(RUSAGE_SELF).ru_maxrss`.

**Why.** CPython does not return freed arenas to the operating system.
Consecutive runs in one interpreter therefore inherit each other's resident
size. Measured on the same BFS run over 20 000 mutants:

| context | reported RSS |
|---|---:|
| alone in a fresh process | 33 MB |
| second run in the same process | 2 666 MB |

An 80× difference, entirely an artifact of run order. The prototype looped over
100 machines and all algorithms in one process, which is consistent with the
constant ~3 426 MB floor visible in its output.

`--no-isolate` runs everything in one process. It is faster — the mutant file
is parsed once per machine instead of once per algorithm — but the memory
columns should not be used from such a run.

### Interval sampling

Metrics are sampled at 30, 60, 90, 120, 150, 180, 240, 300 and 360 seconds.
**Intervals after a run terminates are left empty**, not back-filled.

The prototype back-filled remaining intervals with a reading taken at
back-fill time, so a run that finished at 100 s recorded its 100 s memory in
the "3 min" column. In an earlier BFS result, 58 of 100 machines showed a
memory *drop* of more than 100 MB between consecutive intervals — the moment
the run ended. `Terminated At (sec)` and `Stop Reason` carry that information
instead, and `report.py` averages only over runs still active at each
checkpoint, printing how many contributed.

---

## 5. Shared configuration

| decision | value | note |
|---|---|---|
| initial state | `min(fsm.keys())` | one definition, in `Spec.initial`. The prototype used `next(iter(fsm))` in BFS/DFS and `min(...)` in RL/LA, which can differ. |
| mutant order | file order | identical for all algorithms |
| depth cap | `⌈n/3⌉` (`4` for Černý) | look-ahead runs uncapped; with a cap it usually terminates before finding a solution |
| time budget | 360 s | `--budget` |
| seed | 42 | recorded in every row; the prototype seeded only RL |

---

## 6. Representation

Mutants are stored as a **sparse delta** against the specification — only the
transitions that differ, typically 1–3 per mutant. For 20 000 mutants of a
32-state machine this is 0.6 MB.

The search carries only **surviving** mutants as two `int32` arrays, so its
footprint shrinks as mutants die. The prototype held a fixed-length list of
50 000 entries padded with `None` on every frontier node — 400 000 bytes per
node regardless of how many mutants were still alive. At a measured peak of
41 829 nodes that is ~16 GB, and the implied bytes-per-node in the earlier
results (413 568) matches `50 000 × 8` almost exactly.

`--node-storage replay` stores only the input path per node and recomputes
survivors on expansion. `tests/test_algorithms.py::test_node_storage_modes_agree`
asserts it produces identical results.

---

## Verifying any result

```bash
python -m fsm_mutation.validate results/complete_50000_output.xlsx
```

Each row's `Best Sequence` is replayed against the mutant file by the plain
dictionary interpreter in `validate.py`, which shares no code with the search —
no numpy, no sparse deltas, no shared kill function. Agreement means the two
independent implementations produced the same number. This check runs in CI.
