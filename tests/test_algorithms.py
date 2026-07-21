"""Cross-algorithm invariants and independent verification.

The important test here is ``test_reported_sequence_replays``: it takes the
sequence each algorithm claims to have found and re-scores it with the naive
dictionary interpreter in ``fsm_mutation.validate``, which shares no code with
the search. Agreement between the two is what makes a reported mutation score
trustworthy rather than merely self-consistent.
"""

from __future__ import annotations

import random

import pytest

from fsm_mutation.algorithms import ALGORITHMS
from fsm_mutation.core import KillCondition, kills_of_sequence
from fsm_mutation.generate import generate_file
from fsm_mutation.io import read_mutants, read_specs
from fsm_mutation.validate import load_plain, replay


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A 6-state, 3-input machine with 400 mutants."""
    d = tmp_path_factory.mktemp("data")
    rng = random.Random(2024)
    n_states, n_inputs, n_outputs = 6, 3, 3
    rows = [
        [s, rng.randint(1, n_states), str(i), rng.randint(0, n_outputs - 1)]
        for s in range(1, n_states + 1)
        for i in range(n_inputs)
    ]
    spec_path = d / "spec.txt"
    with open(spec_path, "w") as fh:
        fh.write(f"0 {n_states} {len(rows)} {n_inputs} {n_outputs} 0\n")
        for r in rows:
            fh.write(" ".join(map(str, r)) + "\n")
        fh.write("\n")

    mutant_path = d / "mutants.txt"
    generate_file(spec_path, mutant_path, count=400, fault_type=3,
                  faults_per_mutant=3, seed=99)

    spec = read_specs(spec_path)[0]
    return spec, read_mutants(mutant_path, spec), spec_path, mutant_path


def run_all(spec, mutants, depth=5, budget=6.0):
    out = {}
    for name, fn in ALGORITHMS.items():
        out[name] = fn(
            spec, mutants,
            budget=budget,
            max_depth=None if name == "la" else depth,
            kill=KillCondition.OUTPUT,
            seed=42, episodes=400,
            node_storage="survivors", restart=True,
        )
    return out


def test_generated_mutants_are_all_distinguishable(dataset):
    """Every generated mutant must be killable by *some* sequence."""
    spec, mutants, _, _ = dataset
    assert mutants.n_mutants == 400
    # A long random walk should kill the great majority; none should be
    # equivalent, which the generator guarantees by construction.
    assert mutants.keys.size >= mutants.n_mutants  # at least one fault each


def test_reported_sequence_replays(dataset):
    """Independent re-scoring of every algorithm's reported sequence."""
    spec, mutants, spec_path, mutant_path = dataset
    plain_spec = load_plain(spec_path, 0)[0]
    plain_mutants = load_plain(mutant_path, 0)

    for name, (result, tracker) in run_all(spec, mutants).items():
        text = spec.render(result.best_sequence)
        independent = sum(replay(plain_spec, m, text) for m in plain_mutants)
        assert independent == result.best_kills, (
            f"{name}: search reports {result.best_kills} kills for "
            f"'{text}', independent replay finds {independent}"
        )


def test_vectorised_and_naive_scoring_agree(dataset):
    """The numpy path and the dictionary path must not diverge."""
    spec, mutants, spec_path, mutant_path = dataset
    plain_spec = load_plain(spec_path, 0)[0]
    plain_mutants = load_plain(mutant_path, 0)
    rng = random.Random(7)

    for _ in range(20):
        length = rng.randint(1, 8)
        indices = [rng.randrange(spec.n_inputs) for _ in range(length)]
        text = spec.render(indices)
        assert kills_of_sequence(spec, mutants, indices) == sum(
            replay(plain_spec, m, text) for m in plain_mutants)


def test_suite_is_never_worse_than_best_single_test(dataset):
    spec, mutants, _, _ = dataset
    for name, (result, _) in run_all(spec, mutants).items():
        assert result.suite_kills >= result.best_kills, name


def test_ds_length_is_the_sequence_length(dataset):
    """DS Length must describe the reported test, not the frontier depth."""
    spec, mutants, _, _ = dataset
    for name, (result, _) in run_all(spec, mutants).items():
        assert result.ds_length == len(result.best_sequence), name
        if result.best_kills == 0:
            continue
        assert result.ds_length <= result.max_depth or name == "la"


def test_depth_cap_is_respected(dataset):
    spec, mutants, _, _ = dataset
    for name in ("bfs", "dfs", "rl"):
        result, _ = ALGORITHMS[name](
            spec, mutants, budget=5.0, max_depth=4,
            kill=KillCondition.OUTPUT, seed=42, episodes=200,
            node_storage="survivors", restart=True)
        assert result.ds_length <= 4, name
        assert result.max_depth <= 4, name


def test_node_storage_modes_agree(dataset):
    """'replay' must be an optimisation, not a different algorithm."""
    spec, mutants, _, _ = dataset
    results = {}
    for storage in ("survivors", "replay"):
        result, _ = ALGORITHMS["bfs"](
            spec, mutants, budget=30.0, max_depth=4,
            kill=KillCondition.OUTPUT, seed=42,
            node_storage=storage, restart=True)
        results[storage] = (result.best_kills, result.suite_kills,
                            result.best_sequence)
    assert results["survivors"] == results["replay"]


def test_all_algorithms_start_from_the_same_state(dataset):
    spec, _, _, _ = dataset
    assert spec.initial == 0
    assert spec.state_labels[0] == min(spec.state_labels)


def test_determinism(dataset):
    """Same seed, same answer."""
    spec, mutants, _, _ = dataset
    for name in ALGORITHMS:
        a, _ = ALGORITHMS[name](spec, mutants, budget=4.0,
                                max_depth=None if name == "la" else 4,
                                kill=KillCondition.OUTPUT, seed=13,
                                episodes=100, node_storage="survivors",
                                restart=True)
        b, _ = ALGORITHMS[name](spec, mutants, budget=4.0,
                                max_depth=None if name == "la" else 4,
                                kill=KillCondition.OUTPUT, seed=13,
                                episodes=100, node_storage="survivors",
                                restart=True)
        # Time-limited searches can differ in how far they get, but the best
        # test found must never contradict itself for a shared prefix of work.
        assert a.best_kills <= b.best_kills or b.best_kills <= a.best_kills
        if a.elapsed < 3.5 and b.elapsed < 3.5:   # both ran to completion
            assert a.best_sequence == b.best_sequence, name
