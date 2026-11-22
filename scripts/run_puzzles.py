"""Report how many puzzle positions (V3, #73) the decision engine solves.

    python scripts/run_puzzles.py

For each puzzle: best response's choice with the puzzle's own forecast and with a flat 1/3
forecast, and the probability the minimax mixture puts on accepted answers. Writes
reports/results/puzzles.json.
"""

import json

from pokeagent.config import REPORTS_DIR
from pokeagent.eval.puzzles import FLAT, PUZZLES, solve

REPORT_PATH = REPORTS_DIR / "results" / "puzzles.json"


def main() -> None:
    rows = []
    print(f"{'puzzle':32} {'accepted':28} {'best response':16} {'flat 1/3':16} {'LP mass':>7}")
    for puzzle in PUZZLES:
        own, flat = solve(puzzle), solve(puzzle, FLAT)
        rows.append(
            {
                "puzzle": puzzle.name,
                "idea": puzzle.idea,
                "accepted": sorted(puzzle.accepted),
                "best_response": own["best_response"],
                "solved": own["best_response_solved"],
                "flat_best_response": flat["best_response"],
                "flat_solved": flat["best_response_solved"],
                "lp_accepted_mass": own["lp_accepted_mass"],
                "expected": own["expected"],
            }
        )
        mark = "ok " if own["best_response_solved"] else "NO "
        flat_mark = "ok " if flat["best_response_solved"] else "NO "
        accepted = ",".join(sorted(puzzle.accepted))
        print(
            f"{puzzle.name:32} {accepted[:28]:28} {mark + own['best_response']:16}"
            f" {flat_mark + flat['best_response']:16} {own['lp_accepted_mass']:7.2f}"
        )
    solved = sum(row["solved"] for row in rows)
    flat_solved = sum(row["flat_solved"] for row in rows)
    lp = sum(row["lp_accepted_mass"] for row in rows)
    print(
        f"\nBest response solves {solved}/{len(rows)} (flat forecast {flat_solved}/{len(rows)});"
        f" minimax puts {lp / len(rows):.0%} of its probability on accepted answers on average."
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {"solved": solved, "flat_solved": flat_solved, "puzzles": rows}
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
