"""Independent verification of reported results.

Every row in a results workbook carries the actual input sequence the algorithm
found. This module replays that sequence against the mutant file using a plain,
dictionary-based interpreter that shares no code with the search -- no numpy,
no sparse deltas, no shared kill function -- and checks that the mutation score
comes out the same.

If a scoring bug is ever reintroduced, this catches it, because the two
implementations would have to be wrong in exactly the same way to agree.

Usage::

    python -m fsm_mutation.validate results/complete_50000_output.xlsx \
        --data-dir data
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def load_plain(path, fsm_id):
    """Read every block for ``fsm_id`` as ``{state: {input: (next, out)}}``."""
    blocks = []
    current = None
    current_id = None
    with open(path) as fh:
        for raw in fh:
            parts = raw.split()
            if not parts:
                continue
            if len(parts) == 6 and parts[0].lstrip("-").isdigit():
                if current is not None and current_id == fsm_id:
                    blocks.append(current)
                current_id = int(parts[0])
                current = defaultdict(dict) if current_id == fsm_id else None
            elif len(parts) >= 4 and current is not None:
                src, dst, inp, out = parts[:4]
                current[int(src)][inp] = (int(dst), int(out))
    if current is not None and current_id == fsm_id:
        blocks.append(current)
    return blocks


def replay(spec, mutant, sequence):
    """True if ``sequence`` produces different outputs on ``spec`` and ``mutant``.

    Deliberately naive: one dictionary lookup at a time, comparing outputs and
    nothing else.
    """
    s = min(spec)
    m = min(spec)
    for symbol in sequence:
        if symbol not in spec.get(s, {}):
            return False  # not applicable to the specification
        s_next, s_out = spec[s][symbol]
        if m not in mutant or symbol not in mutant[m]:
            return True   # mutant is silent where the specification speaks
        m_next, m_out = mutant[m][symbol]
        if m_out != s_out:
            return True
        s, m = s_next, m_next
    return False


def verify(results_path, data_dir: Path, tolerance: float = 1e-6) -> int:
    import pandas as pd

    # 'Best Sequence' must be read as text. Left to infer its own dtype,
    # pandas turns a sequence such as '021211' into the integer 21211 and
    # silently drops the leading input.
    df = pd.read_excel(results_path, dtype={"Best Sequence": str})
    failures = 0

    for _, row in df.iterrows():
        fsm_id = int(row["Machine #"])
        sequence = row["Best Sequence"]
        sequence = "" if pd.isna(sequence) else str(sequence)
        reported = float(row["Final Mutation Score (%)"])

        spec_file = data_dir / "fsms" / f"{row['Machine Class']}.txt"
        candidates = list((data_dir / "fsms").glob(f"*{row['Machine Class']}*"))
        if candidates:
            spec_file = candidates[0]
        mutant_file = data_dir / "mutants" / str(row["Mutant File"])
        if not spec_file.exists() or not mutant_file.exists():
            print(f"machine {fsm_id}: source files missing, skipped")
            continue

        specs = load_plain(spec_file, fsm_id)
        if not specs:
            print(f"machine {fsm_id}: specification not found")
            failures += 1
            continue
        spec = specs[0]
        mutants = load_plain(mutant_file, fsm_id)

        killed = sum(replay(spec, m, sequence) for m in mutants)
        actual = killed / len(mutants) * 100 if mutants else 0.0

        ok = abs(actual - reported) < tolerance
        status = "ok" if ok else "MISMATCH"
        if not ok:
            failures += 1
        print(f"machine {fsm_id:<4} {row['Algorithm']:<16} "
              f"len {len(sequence):<4} reported {reported:7.3f}%  "
              f"replayed {actual:7.3f}%  {status}")

    print(f"\n{len(df) - failures}/{len(df)} rows verified")
    return failures


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fsm_mutation.validate")
    p.add_argument("results", type=Path)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args(argv)
    return 1 if verify(args.results, args.data_dir) else 0


if __name__ == "__main__":
    raise SystemExit(main())
