#!/usr/bin/env python3
"""Turn the experiment results into report material.

  uv run python scripts/make_report.py                                   # tables (+ chart if matplotlib is installed)
  uv run python scripts/make_report.py --demo llm,classical              # seeds where the LLM won and classical did not
Outputs go to results/report/: tables.md, win_rates.png (optional) and demo_candidates.md (with --demo).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strategies.llm_commander.reporting import demo_candidates, read_rows, summarise, table_markdown  # noqa: E402


def draw_chart(summary: list[dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    groups = sorted({(s["opponent"], s["observability"]) for s in summary})
    conds = sorted({s["condition"] for s in summary})
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(groups) + 2), 4.2))
    width = 0.8 / max(1, len(conds))
    for i, c in enumerate(conds):
        xs, ys, err = [], [], []
        for j, g in enumerate(groups):
            row = next((s for s in summary if s["condition"] == c and (s["opponent"], s["observability"]) == g), None)
            if row:
                lo, hi = (float(v) for v in row["win_rate_ci95"].split("-"))
                xs.append(j + i * width)
                ys.append(row["win_rate"] * 100)
                err.append([max(0, row["win_rate"] * 100 - lo * 100), max(0, hi * 100 - row["win_rate"] * 100)])
        ax.bar(xs, ys, width, label=c, yerr=list(zip(*err)) if err else None, capsize=2)
    ax.set_xticks([j + width * (len(conds) - 1) / 2 for j in range(len(groups))])
    ax.set_xticklabels([f"{o}\n({m})" for o, m in groups])
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.set_title("Win rate by condition (95% confidence intervals)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", default="results/games.csv")
    p.add_argument("--out", default="results/report")
    p.add_argument("--demo", default=None, help="A,B: list games where condition A won and B did not (and the reverse)")
    args = p.parse_args()

    rows = read_rows(ROOT / args.games)
    if not rows:
        sys.exit(f"No results in {args.games}. Run scripts/run_experiments.py first.")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(rows)
    (out / "tables.md").write_text(table_markdown(summary))
    print(f"Wrote {out / 'tables.md'}")
    print(f"Wrote {out / 'win_rates.png'}" if draw_chart(summary, out / "win_rates.png")
          else "matplotlib not installed: skipped the chart (uv run --with matplotlib ...)")
    if args.demo:
        a, b = args.demo.split(",")
        better, worse = demo_candidates(rows, a, b)
        lines = [f"# Demo candidates: {a} vs {b}\n", f"## {a} won, {b} did not ({len(better)})\n"]
        lines += [f"- {d}" for d in better] or ["- none"]
        lines += [f"\n## {a} did not win, {b} won: failure cases ({len(worse)})\n"] + ([f"- {d}" for d in worse] or ["- none"])
        (out / "demo_candidates.md").write_text("\n".join(lines) + "\n")
        print(f"Wrote {out / 'demo_candidates.md'} ({len(better)} demo candidates, {len(worse)} failure cases)")
        for d in better[:3]:
            print(f"  demo: --opponent {d['opponent']} --team {d['side']} --seed {d['seed']} --spawn near_base"
                  + (" --partial" if d["observability"] == "partial" else ""))


if __name__ == "__main__":
    main()
