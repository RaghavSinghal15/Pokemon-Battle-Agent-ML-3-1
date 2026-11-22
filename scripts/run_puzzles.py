"""Report how many puzzle positions (V3, #73) the decision engine solves.

    python scripts/run_puzzles.py

For each puzzle: best response's choice with the puzzle's own forecast and with a flat 1/3
forecast, and the probability the minimax mixture puts on accepted answers. If the logistic
value model has been trained (python -m pokeagent.models.train_value), also best response
with the value-model utility (V4). Writes reports/results/puzzles.json.
"""

import json
import pickle

from pokeagent.config import REPORTS_DIR, VALUE_UTILITY_MODEL_PATH
from pokeagent.eval.puzzles import FLAT, PUZZLES, solve

REPORT_PATH = REPORTS_DIR / "results" / "puzzles.json"


def main() -> None:
    value_model = None
    if VALUE_UTILITY_MODEL_PATH.is_file():
        with VALUE_UTILITY_MODEL_PATH.open("rb") as stream:
            value_model = pickle.load(stream)
    rows = []
    print(
        f"{'puzzle':32} {'accepted':28} {'best response':16} {'flat 1/3':16} {'LP mass':>7}"
        f"  {'value utility' if value_model else ''}"
    )
    for puzzle in PUZZLES:
        own, flat = solve(puzzle), solve(puzzle, FLAT)
        value = solve(puzzle, value_model=value_model) if value_model else None
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
                "value_best_response": value and value["best_response"],
                "value_solved": value and value["best_response_solved"],
                "value_expected": value and value["expected"],
            }
        )
        mark = "ok " if own["best_response_solved"] else "NO "
        flat_mark = "ok " if flat["best_response_solved"] else "NO "
        accepted = ",".join(sorted(puzzle.accepted))
        print(
            f"{puzzle.name:32} {accepted[:28]:28} {mark + own['best_response']:16}"
            f" {flat_mark + flat['best_response']:16} {own['lp_accepted_mass']:7.2f}"
            + (
                f"  {'ok ' if value['best_response_solved'] else 'NO '}{value['best_response']}"
                if value
                else ""
            )
        )
    solved = sum(row["solved"] for row in rows)
    flat_solved = sum(row["flat_solved"] for row in rows)
    lp = sum(row["lp_accepted_mass"] for row in rows)
    print(
        f"\nBest response solves {solved}/{len(rows)} (flat forecast {flat_solved}/{len(rows)});"
        f" minimax puts {lp / len(rows):.0%} of its probability on accepted answers on average."
    )
    value_solved = sum(bool(row["value_solved"]) for row in rows) if value_model else None
    if value_model:
        print(f"With the value-model utility, best response solves {value_solved}/{len(rows)}.")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "solved": solved,
        "flat_solved": flat_solved,
        "value_solved": value_solved,
        "puzzles": rows,
    }
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
