"""Aggregate raw per-machine rows into the report workbook.

One sheet per machine class, each containing the three views the paper uses:

1. Average mutation score vs. time, one series per algorithm
2. Average mutation score and average DS length, per machine
3. Average memory (MB) vs. time, one series per algorithm

Averages skip machines whose run had already terminated at a given checkpoint,
so a mean is always taken over runs that were actually still going. The count
of contributing runs is written beside every row -- an average over 3 machines
and an average over 100 are not the same claim, and the reader should be able
to tell them apart.

Usage::

    python -m fsm_mutation.report results/*.xlsx -o results/report.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from .core import INTERVAL_NAMES

TITLE = Font(bold=True, size=14)
HEADING = Font(bold=True, size=11)
MINUTES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]


def load(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_excel(p, dtype={"Best Sequence": str})
        df["Source File"] = Path(p).name
        frames.append(df)
    if not frames:
        raise SystemExit("no input workbooks")
    return pd.concat(frames, ignore_index=True)


def _series(df: pd.DataFrame, template: str) -> tuple:
    """Mean of an interval metric per algorithm, plus contributing run counts."""
    cols = [template.format(n) for n in INTERVAL_NAMES]
    cols = [c for c in cols if c in df.columns]
    means, counts, algorithms = {}, {}, sorted(df["Algorithm"].unique())
    for alg in algorithms:
        sub = df[df["Algorithm"] == alg][cols]
        means[alg] = [sub[c].mean(skipna=True) for c in cols]
        counts[alg] = [int(sub[c].notna().sum()) for c in cols]
    return algorithms, means, counts


def _write_block(ws, row: int, title: str, algorithms, means, counts,
                 number_format: str = "0.00") -> int:
    ws.cell(row=row, column=1, value=title).font = HEADING
    row += 1

    ws.cell(row=row, column=1, value="Algorithm").font = HEADING
    for j, name in enumerate(INTERVAL_NAMES):
        c = ws.cell(row=row, column=2 + j, value=name)
        c.font = HEADING
        c.alignment = Alignment(horizontal="center")
    ws.cell(row=row, column=2 + len(INTERVAL_NAMES), value="runs (min-max)").font = HEADING
    header_row = row
    row += 1

    first_data = row
    for alg in algorithms:
        ws.cell(row=row, column=1, value=alg)
        for j, v in enumerate(means[alg]):
            c = ws.cell(row=row, column=2 + j,
                        value=None if v is None or np.isnan(v) else float(v))
            c.number_format = number_format
        n = counts[alg]
        ws.cell(row=row, column=2 + len(INTERVAL_NAMES),
                value=f"{min(n)}-{max(n)}" if n else "0")
        row += 1
    return header_row, first_data, row


def _add_line_chart(ws, anchor: str, title: str, y_title: str,
                    header_row: int, first_data: int, last_data: int) -> None:
    chart = LineChart()
    chart.title = title
    chart.y_axis.title = y_title
    chart.x_axis.title = "Execution time (min)"
    chart.height, chart.width = 8, 17
    data = Reference(ws, min_col=2, max_col=1 + len(INTERVAL_NAMES),
                     min_row=first_data, max_row=last_data)
    cats = Reference(ws, min_col=2, max_col=1 + len(INTERVAL_NAMES),
                     min_row=header_row)
    chart.add_data(data, from_rows=True, titles_from_data=False)
    chart.set_categories(cats)
    for i, s in enumerate(chart.series):
        s.smooth = False
        label = ws.cell(row=first_data + i, column=1).value
        s.tx = None
    ws.add_chart(chart, anchor)


def build(df: pd.DataFrame, out_path) -> None:
    wb = Workbook()
    wb.remove(wb.active)

    for machine_class, sub in df.groupby("Machine Class"):
        ws = wb.create_sheet(str(machine_class)[:31])
        ws.column_dimensions["A"].width = 24
        for j in range(len(INTERVAL_NAMES) + 1):
            ws.column_dimensions[get_column_letter(2 + j)].width = 12

        mutant_counts = sorted(sub["Number of Mutants"].unique())
        kill = ", ".join(sorted(sub["Kill Condition"].astype(str).unique()))
        ws["A1"] = (f"{machine_class} — "
                    f"{sub['Machine #'].nunique()} machines, "
                    f"{'/'.join(str(m) for m in mutant_counts)} mutants, "
                    f"kill condition: {kill}")
        ws["A1"].font = TITLE
        row = 3

        # 1) Average mutation score vs time -------------------------------
        algs, means, counts = _series(sub, "Mutation Score after {} (%)")
        hdr, first, row = _write_block(
            ws, row, "1) Average Mutation Score vs Time "
                     "(best single distinguishing test)", algs, means, counts)
        _add_line_chart(ws, f"A{row + 1}",
                        "Average mutation score vs time",
                        "Mutation score (%)", hdr, first, row - 1)
        row += 18

        algs, means, counts = _series(sub, "Suite Score after {} (%)")
        hdr, first, row = _write_block(
            ws, row, "1b) Average Suite Score vs Time "
                     "(union over every sequence explored)", algs, means, counts)
        _add_line_chart(ws, f"A{row + 1}",
                        "Average suite score vs time",
                        "Suite score (%)", hdr, first, row - 1)
        row += 18

        # 2) Score and DS length per machine ------------------------------
        ws.cell(row=row, column=1,
                value="2) Average Mutation Score and Average DS Length "
                      "per machine").font = HEADING
        row += 1
        per = (sub.groupby(["Machine #", "Algorithm"])
                  .agg(score=("Final Mutation Score (%)", "mean"),
                       ds=("Final DS Length", "mean"),
                       suite=("Suite Score (%)", "mean"),
                       cost=("Suite Cost (# inputs)", "mean"))
                  .reset_index())
        headers = ["Machine #", "Algorithm", "Avg Mutation Score (%)",
                   "Avg DS Length", "Avg Suite Score (%)", "Avg Suite Cost"]
        for j, h in enumerate(headers):
            ws.cell(row=row, column=1 + j, value=h).font = HEADING
        row += 1
        for _, r in per.iterrows():
            ws.cell(row=row, column=1, value=int(r["Machine #"]))
            ws.cell(row=row, column=2, value=str(r["Algorithm"]))
            ws.cell(row=row, column=3, value=float(r["score"])).number_format = "0.00"
            ws.cell(row=row, column=4, value=float(r["ds"])).number_format = "0.0"
            ws.cell(row=row, column=5, value=float(r["suite"])).number_format = "0.00"
            ws.cell(row=row, column=6, value=float(r["cost"])).number_format = "0"
            row += 1
        row += 2

        # 3) Average memory vs time ---------------------------------------
        algs, means, counts = _series(sub, "Memory Usage (MB) after {}")
        hdr, first, row = _write_block(
            ws, row, "3) Average Memory (MB) vs Time (min)", algs, means, counts,
            number_format="0")
        _add_line_chart(ws, f"A{row + 1}", "Average memory vs time",
                        "Memory (MB)", hdr, first, row - 1)
        row += 18

        ws.cell(row=row, column=1, value=(
            "Averages exclude runs that had already terminated at a "
            "checkpoint; 'runs' shows how many contributed."))

    # Flat sheet with everything, for anyone who wants to pivot it themselves.
    raw = wb.create_sheet("raw")
    raw.append(list(df.columns))
    for r in df.itertuples(index=False):
        raw.append([None if (isinstance(v, float) and np.isnan(v)) else v
                    for v in r])

    wb.save(out_path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fsm_mutation.report")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("results/report.xlsx"))
    args = p.parse_args(argv)

    df = load(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(df, args.output)
    print(f"wrote {args.output} "
          f"({df['Machine Class'].nunique()} sheets, {len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
