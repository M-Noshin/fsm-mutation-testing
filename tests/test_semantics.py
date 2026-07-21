"""Kill semantics, checked against a machine small enough to verify by hand.

Specification (states 1-2, inputs 0-1, outputs 0-1)::

    1 --0/0--> 2        1 --1/1--> 1
    2 --0/1--> 1        2 --1/0--> 2

Three mutants, each with one fault:

=====  ======================================  ==============================
  A    output of (1, 0) becomes 1              killed by "0"
  B    next state of (1, 0) becomes 1          killed by "00", not by "0"
  C    output of (2, 1) becomes 1              killed by "01"
=====  ======================================  ==============================

Mutant B is the interesting one: it is a pure transfer fault, so nothing is
observable at the step where it is traversed. A tester applying "0" sees output
0 from both machines. Only after a second input does the divergent state emit a
different output.
"""

from __future__ import annotations

import textwrap

import pytest

from fsm_mutation.core import KillCondition, kills_of_sequence
from fsm_mutation.io import read_mutants, read_specs

SPEC = """\
0 2 4 2 2 0
1 2 0 0
1 1 1 1
2 1 0 1
2 2 1 0
"""

# A: output fault at (1,0)   B: transfer fault at (1,0)   C: output fault at (2,1)
MUTANTS = """\
0 2 4 2 2 0
1 2 0 1
1 1 1 1
2 1 0 1
2 2 1 0

0 2 4 2 2 0
1 1 0 0
1 1 1 1
2 1 0 1
2 2 1 0

0 2 4 2 2 0
1 2 0 0
1 1 1 1
2 1 0 1
2 2 1 1
"""

A, B, C = 0, 1, 2


@pytest.fixture
def machine(tmp_path):
    spec_path = tmp_path / "spec.txt"
    mutant_path = tmp_path / "mutants.txt"
    spec_path.write_text(SPEC)
    mutant_path.write_text(MUTANTS)
    spec = read_specs(spec_path)[0]
    return spec, read_mutants(mutant_path, spec)


def seq(spec, text):
    """Turn '00' into a list of input indices."""
    return [spec.input_labels.index(ch) for ch in text]


def test_parsing(machine):
    spec, mutants = machine
    assert spec.n_states == 2
    assert spec.n_inputs == 2
    assert spec.n_transitions == 4
    assert spec.initial == 0            # label 1, the smallest
    assert mutants.n_mutants == 3
    assert mutants.keys.size == 3       # exactly one changed transition each


def test_output_fault_killed_immediately(machine):
    spec, mutants = machine
    assert kills_of_sequence(spec, mutants, seq(spec, "0")) == 1  # only A


def test_transfer_fault_is_not_observable_at_the_faulty_step(machine):
    """The core correction: a state difference alone is not a kill."""
    spec, mutants = machine
    correct = kills_of_sequence(spec, mutants, seq(spec, "0"),
                                KillCondition.OUTPUT)
    legacy = kills_of_sequence(spec, mutants, seq(spec, "0"),
                               KillCondition.OUTPUT_OR_STATE)
    assert correct == 1   # A only
    assert legacy == 2    # A and B -- B counted although nothing is observable


def test_transfer_fault_killed_once_it_shows(machine):
    spec, mutants = machine
    assert kills_of_sequence(spec, mutants, seq(spec, "00")) == 2  # A and B


def test_shortest_sequence_killing_all_three(machine):
    """"01" kills every mutant; nothing of length 1 does.

    B falls to this sequence as well as C: after the prefix "0" the transfer
    fault has put B in state 1 while the specification is in state 2, and
    input 1 then yields output 1 from B against 0 from the specification. The
    kill is real, but it happens one step *after* the fault was traversed.
    """
    spec, mutants = machine
    assert kills_of_sequence(spec, mutants, seq(spec, "01")) == 3
    assert kills_of_sequence(spec, mutants, seq(spec, "1")) == 0
    assert kills_of_sequence(spec, mutants, seq(spec, "0")) == 1


# --------------------------------------------------------------------------
# A machine where no single test can kill everything.
# --------------------------------------------------------------------------
#
#   1 --0/0--> 2      1 --1/0--> 3      2 and 3 self-loop on both inputs.
#
# Once the first input is chosen the other branch is unreachable forever, so a
# fault on (1,0) and a fault on (1,1) cannot both be exercised by one sequence.

FORK_SPEC = """\
1 3 6 2 2 0
1 2 0 0
1 3 1 0
2 2 0 0
2 2 1 0
3 3 0 0
3 3 1 0
"""

FORK_MUTANTS = """\
1 3 6 2 2 0
1 2 0 1
1 3 1 0
2 2 0 0
2 2 1 0
3 3 0 0
3 3 1 0

1 3 6 2 2 0
1 2 0 0
1 3 1 1
2 2 0 0
2 2 1 0
3 3 0 0
3 3 1 0
"""


def test_suite_beats_any_single_test(tmp_path):
    """The suite kills 100%, the best single test kills 50%.

    Reporting the union of kills over a search tree as though it were the score
    of one distinguishing test overstates by a factor of two here, and by much
    more on a machine with a wide input alphabet.
    """
    from itertools import product

    (tmp_path / "s.txt").write_text(FORK_SPEC)
    (tmp_path / "m.txt").write_text(FORK_MUTANTS)
    spec = read_specs(tmp_path / "s.txt")[0]
    mutants = read_mutants(tmp_path / "m.txt", spec)
    assert mutants.n_mutants == 2

    best = max(
        kills_of_sequence(spec, mutants, list(combo))
        for length in range(1, 6)
        for combo in product(range(spec.n_inputs), repeat=length)
    )
    assert best == 1                       # 50% -- the honest figure

    union = set()
    for length in range(1, 6):
        for combo in product(range(spec.n_inputs), repeat=length):
            if kills_of_sequence(spec, mutants, list(combo)):
                union.add(combo[0])
    assert len(union) == 2                 # 100% -- what cumulative counting reports
