"""Result tables for the report: win rates with confidence intervals, demo/failure seed candidates."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["opponent"], r["observability"])].append(r)
    out = []
    num = lambda r, k: float(r.get(k) or 0)  # noqa: E731
    for (cond, opp, obs), rs in sorted(groups.items()):
        n = len(rs)
        wins = sum(r["result"] == "win" for r in rs)
        lo, hi = wilson(wins, n)
        proposed = sum(num(r, "orders") + num(r, "illegal_orders") for r in rs)
        reqs = sum(num(r, "llm_requests") for r in rs)
        out.append({
            "condition": cond, "opponent": opp, "observability": obs, "games": n, "wins": wins,
            "losses": sum(r["result"] == "loss" for r in rs), "draws": sum(r["result"] == "draw" for r in rs),
            "win_rate": round(wins / n, 3), "win_rate_ci95": f"{lo:.2f}-{hi:.2f}",
            "avg_turns": round(sum(num(r, "turns") for r in rs) / n, 1),
            "avg_flag_pickups": round(sum(num(r, "my_flag_pickups") for r in rs) / n, 2),
            "avg_agents_tagged": round(sum(num(r, "my_agents_tagged") for r in rs) / n, 2),
            "avg_moves": round(sum(num(r, "my_moves") for r in rs) / n, 1),
            "illegal_order_rate": round(sum(num(r, "illegal_orders") for r in rs) / proposed, 3) if proposed else "",
            "avg_llm_seconds_per_request": round(sum(num(r, "llm_seconds") for r in rs) / reqs, 2) if reqs else "",
            "tokens_in": int(sum(num(r, "tokens_in") for r in rs)), "tokens_out": int(sum(num(r, "tokens_out") for r in rs)),
        })
    return out


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))




def table_markdown(summary: list[dict]) -> str:
    """One Markdown table per opponent / observability setting."""
    blocks = defaultdict(list)
    for s in summary:
        blocks[(s["opponent"], s["observability"])].append(s)
    out = []
    for (opp, obs), rows in sorted(blocks.items()):
        out.append(f"### vs {opp} ({obs} observability)\n")
        out.append("| Condition | Games | Win % | 95% CI | Draws | Avg turns | Flag pickups | Agents tagged | Illegal-order rate | s/request |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            out.append(f"| {r['condition']} | {r['games']} | {r['win_rate'] * 100:.0f} | {r['win_rate_ci95']} | {r['draws']} | "
                       f"{r['avg_turns']} | {r['avg_flag_pickups']} | {r['avg_agents_tagged']} | "
                       f"{r['illegal_order_rate']} | {r['avg_llm_seconds_per_request']} |")
        out.append("")
    return "\n".join(out)


def demo_candidates(rows: list[dict], a: str, b: str) -> tuple[list[dict], list[dict]]:
    """Games (same opponent, observability, side, seed) where condition a won and b did not (demo
    candidates), and where a did not win but b did (failure cases for the report)."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["opponent"], r["observability"], r["side"], r["seed"])][r["condition"]] = r
    better, worse = [], []
    for key, d in sorted(by.items()):
        if a in d and b in d:
            ra, rb = d[a], d[b]
            item = {"opponent": key[0], "observability": key[1], "side": key[2], "seed": key[3],
                    f"{a}_result": ra["result"], f"{a}_turns": ra["turns"],
                    f"{b}_result": rb["result"], f"{b}_turns": rb["turns"]}
            if ra["result"] == "win" and rb["result"] != "win":
                better.append(item)
            elif ra["result"] != "win" and rb["result"] == "win":
                worse.append(item)
    return better, worse
