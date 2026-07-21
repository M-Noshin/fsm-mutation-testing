"""Single-run worker, executed in its own process.

Memory figures are only meaningful in a process that has done nothing else.
CPython does not return freed arenas to the operating system, so a run that
follows a memory-hungry one inherits its resident size: in testing, a search
measuring 33 MB on its own reported 2,666 MB when it ran second in the same
interpreter. Averaged over a hundred machines that turns the memory column
into a record of whichever machine happened to run first.

The parent process (``fsm_mutation.run``) therefore spawns one of these per
(machine, algorithm) pair and collects the resulting row as JSON on stdout.
"""

from __future__ import annotations

import json
import sys


def main(argv=None) -> int:
    spec = json.loads((argv or sys.argv[1:])[0])

    from .algorithms import ALGORITHMS
    from .core import KillCondition, result_row
    from .io import read_mutants, read_specs

    specs = {s.fsm_id: s for s in read_specs(spec["spec_path"])}
    fsm = specs[spec["fsm_id"]]
    mutants = read_mutants(spec["mutant_path"], fsm, limit=spec["limit"])
    kill = KillCondition(spec["kill"])

    result, tracker = ALGORITHMS[spec["algorithm"]](
        fsm, mutants,
        budget=spec["budget"],
        max_depth=spec["max_depth"],
        kill=kill,
        seed=spec["seed"],
        episodes=spec["episodes"],
        node_storage=spec["node_storage"],
        restart=spec["restart"],
    )
    row = result_row(
        fsm, mutants, result, tracker,
        machine_class=spec["machine_class"],
        mutant_file=spec["mutant_name"],
        max_depth_cap=spec["max_depth"],
        seed=spec["seed"],
        kill=kill,
    )
    sys.stdout.write("\x00JSON\x00" + json.dumps(row, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
