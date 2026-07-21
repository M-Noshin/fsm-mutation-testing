"""Search algorithms, all sharing the semantics defined in ``core``."""

from .lookahead import run_lookahead
from .qlearning import run_qlearning
from .tree import run_bfs, run_dfs

#: Registry used by the CLI. Keys are the ``--algorithm`` values.
ALGORITHMS = {
    "bfs": run_bfs,
    "dfs": run_dfs,
    "rl": run_qlearning,
    "la": run_lookahead,
}

__all__ = ["ALGORITHMS", "run_bfs", "run_dfs", "run_qlearning", "run_lookahead"]
