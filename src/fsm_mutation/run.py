"""Command-line experiment runner.

Examples
--------
Run every algorithm on the complete 128-state machines against 50 000 mutants::

    python -m fsm_mutation.run --machines complete --algorithm all --mutants 50000

Run only Q-learning on the Cerny machines::

    python -m fsm_mutation.run --machines cerny --algorithm rl --mutants 10000

Reproduce the older (incorrect) scoring for comparison::

    python -m fsm_mutation.run --machines complete --algorithm bfs \
        --kill-condition output_or_state
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

from .algorithms import ALGORITHMS
from .core import KillCondition, append_row, result_row
from .io import read_mutants, read_specs

#: Machine classes shipped with the repository. Each maps to a spec file and
#: the depth rule the paper uses for it.
MACHINE_CLASSES = {
    "complete": {"spec": "128state_fsms_complete.txt", "depth": "n/3"},
    "partial": {"spec": "128state_fsms_partial.txt", "depth": "n/3"},
    "cerny": {"spec": "cerny_machines.txt", "depth": "4"},
    "random": {"spec": "random_fsms.txt", "depth": "n/3"},
}


def resolve_depth(rule, n_states: int, algorithm: str):
    """Maximum search depth, or None for uncapped.

    Look-ahead runs uncapped: with a depth cap it usually terminates before
    finding a solution, which is the behaviour reported in the paper.
    """
    if algorithm == "la":
        return None
    if rule in (None, "none"):
        return None
    if rule == "n/3":
        return math.ceil(n_states / 3)
    if rule == "n/2":
        return math.ceil(n_states / 2)
    if rule == "n-1":
        return n_states - 1
    return int(rule)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fsm_mutation.run",
        description="Derive distinguishing tests for FSM mutants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--machines", default="complete",
                   choices=sorted(MACHINE_CLASSES) + ["custom"],
                   help="which family of specifications to run")
    p.add_argument("--spec-file", type=Path,
                   help="explicit spec path (required for --machines custom)")
    p.add_argument("--mutant-file", type=Path,
                   help="explicit mutant path; otherwise derived from "
                        "--machines and --mutants")
    p.add_argument("--algorithm", default="all",
                   choices=sorted(ALGORITHMS) + ["all"])
    p.add_argument("--mutants", type=int, default=50_000,
                   help="mutant count, used to locate the mutant file")
    p.add_argument("--limit-mutants", type=int,
                   help="read at most this many mutants (for quick trials)")
    p.add_argument("--machine", type=int, action="append",
                   help="restrict to these machine ids; repeatable")

    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))

    p.add_argument("--budget", type=float, default=360.0,
                   help="per-machine time budget in seconds")
    p.add_argument("--depth", default=None,
                   help="depth rule: n/3, n/2, n-1, an integer, or none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kill-condition", default="output",
                   choices=[k.value for k in KillCondition],
                   help="'output' is correct; 'output_or_state' reproduces "
                        "the older, inflated scoring")
    p.add_argument("--node-storage", default="survivors",
                   choices=["survivors", "replay"],
                   help="BFS/DFS frontier representation")
    p.add_argument("--episodes", type=int, default=10_000,
                   help="Q-learning episode cap")
    p.add_argument("--no-la-restart", action="store_true",
                   help="stop look-ahead at the first dead end")
    p.add_argument("--no-isolate", action="store_true",
                   help="run everything in one process. Faster, because the "
                        "mutant file is parsed once per machine instead of "
                        "once per algorithm, but the memory columns become "
                        "meaningless: freed memory is not returned to the OS, "
                        "so each run inherits the resident size of the one "
                        "before it.")
    return p


def _run_isolated(payload: dict) -> dict:
    """Execute one run in a fresh interpreter and return its result row."""
    import json
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "fsm_mutation._worker", json.dumps(payload)],
        capture_output=True, text=True,
    )
    marker = "\x00JSON\x00"
    if proc.returncode != 0 or marker not in proc.stdout:
        raise RuntimeError(
            f"worker failed (exit {proc.returncode})\n"
            f"{proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout.split(marker, 1)[1])


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.machines == "custom":
        if not args.spec_file:
            print("--machines custom requires --spec-file", file=sys.stderr)
            return 2
        spec_path = args.spec_file
        depth_rule = args.depth or "n/3"
        label = args.spec_file.stem
    else:
        cfg = MACHINE_CLASSES[args.machines]
        spec_path = args.data_dir / "fsms" / cfg["spec"]
        depth_rule = args.depth or cfg["depth"]
        label = args.machines

    mutant_path = args.mutant_file or (
        args.data_dir / "mutants" / f"{label}_{args.mutants}.txt")

    for path, what in ((spec_path, "specification"), (mutant_path, "mutant")):
        if not Path(path).exists():
            print(f"{what} file not found: {path}", file=sys.stderr)
            return 1

    kill = KillCondition(args.kill_condition)
    algorithms = sorted(ALGORITHMS) if args.algorithm == "all" else [args.algorithm]
    specs = read_specs(spec_path)
    if args.machine:
        wanted = set(args.machine)
        specs = [s for s in specs if s.fsm_id in wanted]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{label}_{args.mutants}_{args.kill_condition}.xlsx"

    print(f"specs      {spec_path}  ({len(specs)} machines)")
    print(f"mutants    {mutant_path}")
    print(f"algorithms {', '.join(algorithms)}")
    print(f"kill       {kill.value}")
    print(f"output     {out}\n")

    isolate = not args.no_isolate
    if not isolate:
        print("warning: --no-isolate makes the memory columns unreliable\n")

    for spec in specs:
        mutants = None
        if not isolate:
            t0 = time.perf_counter()
            mutants = read_mutants(mutant_path, spec, limit=args.limit_mutants)
            if mutants.n_mutants == 0:
                print(f"machine {spec.fsm_id}: no mutants found, skipping")
                continue
            print(f"machine {spec.fsm_id}: {mutants.n_mutants} mutants "
                  f"({mutants.nbytes() / 1024**2:.1f} MB, "
                  f"loaded in {time.perf_counter() - t0:.1f}s)")
        else:
            print(f"machine {spec.fsm_id}:")

        for name in algorithms:
            depth = resolve_depth(depth_rule, spec.n_states, name)
            payload = {
                "spec_path": str(spec_path),
                "mutant_path": str(mutant_path),
                "mutant_name": Path(mutant_path).name,
                "fsm_id": spec.fsm_id,
                "algorithm": name,
                "budget": args.budget,
                "max_depth": depth,
                "kill": kill.value,
                "seed": args.seed,
                "episodes": args.episodes,
                "node_storage": args.node_storage,
                "restart": not args.no_la_restart,
                "limit": args.limit_mutants,
                "machine_class": label,
            }
            if isolate:
                try:
                    row = _run_isolated(payload)
                except RuntimeError as exc:
                    print(f"  {name:<16} FAILED: {exc}")
                    continue
            else:
                result, tracker = ALGORITHMS[name](
                    spec, mutants,
                    budget=args.budget, max_depth=depth, kill=kill,
                    seed=args.seed, episodes=args.episodes,
                    node_storage=args.node_storage,
                    restart=not args.no_la_restart,
                )
                row = result_row(
                    spec, mutants, result, tracker,
                    machine_class=label, mutant_file=Path(mutant_path).name,
                    max_depth_cap=depth, seed=args.seed, kill=kill,
                )

            append_row(out, row)
            print(f"  {row['Algorithm']:<16} "
                  f"DT {float(row['Final Mutation Score (%)']):6.2f}% "
                  f"len {row['Final DS Length']:<4} | "
                  f"suite {float(row['Suite Score (%)']):6.2f}% "
                  f"({row['Suite Size (# tests)']} tests, "
                  f"{row['Suite Cost (# inputs)']} inputs) | "
                  f"peak {float(row['Peak Memory (MB)']):.0f} MB | "
                  f"{float(row['Elapsed Time (sec)']):.1f}s  {row['Stop Reason']}")

        del mutants
        gc.collect()

    print(f"\nwrote {out}")
    print(f"verify with: python -m fsm_mutation.validate {out} "
          f"--data-dir {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
