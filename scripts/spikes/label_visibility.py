"""How often can we see what each player chose? (foundation check 4)

A replay shows what HAPPENED, not what a player CHOSE. For every turn and every player we look
at the public log and put the turn in one bucket:

  switch          the player switched (a switch before any move this turn is a real choice)
  attack          they used a Physical or Special move
  non_damaging    they used a Status move
  cant            "|cant|" line (asleep, fully paralysed, flinched, recharging, ...) or hit itself
                  in confusion: the choice is hidden
  fainted_first   their Pokemon fainted before it acted (choice hidden)
  locked          a move they were locked into ("[from]lockedmove", e.g. Outrage): not a free choice
  battle_ended    the game ended before they acted
  unknown         none of the above (worth inspecting)

The first three are usable labels. Everything else is dropped from opponent-model training.

Usage:
    python scripts/spikes/label_visibility.py data/interim/local_games.jsonl
"""

import argparse
import json
from collections import Counter
from functools import cache
from pathlib import Path

from poke_env.battle import Move, MoveCategory

LABELS = ("switch", "attack", "non_damaging")


@cache
def move_class(move_name: str) -> str:
    move = Move("".join(c for c in move_name.lower() if c.isalnum()), gen=9)
    return "non_damaging" if move.category == MoveCategory.STATUS else "attack"


def side_of(identifier: str) -> str:
    return identifier[:2]  # "p1a: Garchomp" -> "p1"


def turn_blocks(log_lines: list[str]):
    """Yield (turn_number, lines) for each turn: the lines after |turn|N up to the next |turn|."""
    turn, block = None, []
    for line in log_lines:
        if line.startswith("|turn|"):
            if turn is not None:
                yield turn, block
            turn, block = int(line.split("|")[2]), []
        elif turn is not None:
            block.append(line)
    if turn is not None:
        yield turn, block


def classify_turn(block: list[str]) -> dict[str, str]:
    """Bucket each side's choice for one turn."""
    result: dict[str, str] = {}
    any_move_yet = False
    for line in block:
        parts = line.split("|")
        if len(parts) < 3:
            if line.startswith("|upkeep"):
                break  # after upkeep come forced replacements, not choices
            continue
        kind = parts[1]
        if kind == "upkeep":
            break
        side = side_of(parts[2]) if parts[2][:1] == "p" else None
        if side is None or side in result:
            if kind == "move":
                any_move_yet = True
            continue
        if kind == "switch" and not any_move_yet:
            result[side] = "switch"  # switches happen before moves, so this was chosen
        elif kind == "move":
            any_move_yet = True
            if any(p.startswith("[from]") for p in parts[4:]):
                tag = next(p for p in parts[4:] if p.startswith("[from]"))
                if "lockedmove" in tag:
                    result[side] = "locked"
                # other [from] moves (Magic Bounce, Dancer, ...) are not this side's choice
                continue
            result[side] = move_class(parts[3])
        elif kind == "cant":
            result[side] = "cant"
        elif kind == "-damage" and "[from] confusion" in parts[4:]:
            result[side] = "cant"  # hit itself in confusion: no move line, choice hidden
        elif kind == "faint":
            result[side] = "fainted_first"
    for side in ("p1", "p2"):
        if side not in result:
            ended = any(line.startswith(("|win|", "|tie")) for line in block)
            result[side] = "battle_ended" if ended else "unknown"
    return result


def visibility(log_lines: list[str]) -> Counter:
    counts: Counter = Counter()
    for _turn, block in turn_blocks(log_lines):
        for bucket in classify_turn(block).values():
            counts[bucket] += 1
    return counts


def report(counts: Counter, title: str) -> None:
    total = sum(counts.values())
    usable = sum(counts[k] for k in LABELS)
    print(f"{title}: {total} (player, turn) pairs, {usable} usable ({100 * usable / total:.1f}%)")
    for k, v in counts.most_common():
        print(f"  {k:<14} {v:>7}  ({100 * v / total:5.1f}%)")


def true_class(choice: dict) -> str:
    return "switch" if choice["kind"] == "switch" else move_class(choice["name"])


def main(path: Path) -> None:
    by_matchup: dict[str, Counter] = {}
    total: Counter = Counter()
    # Local games also record what each bot REALLY chose, so we can check the log-based labels
    # against ground truth: (label read from the log, true class) -> count.
    confusion: Counter = Counter()
    for line in path.open():
        rec = json.loads(line)
        c = visibility(rec["spectator_log"])
        by_matchup.setdefault(f"{rec['p1_kind']} vs {rec['p2_kind']}", Counter()).update(c)
        total.update(c)
        truth = rec.get("true_choices")
        if truth:
            for turn, block in turn_blocks(rec["spectator_log"]):
                for side, bucket in classify_turn(block).items():
                    choice = truth.get(side, {}).get(str(turn))
                    if choice is not None:
                        confusion[(bucket, true_class(choice))] += 1
    for name, c in by_matchup.items():
        report(c, name)
    report(total, "ALL")
    if confusion:
        usable = {k: v for k, v in confusion.items() if k[0] in LABELS}
        correct = sum(v for (a, b), v in usable.items() if a == b)
        print(f"\nGround truth: usable labels correct {correct}/{sum(usable.values())}")
        print("  (label from log, true class): count")
        for k, v in sorted(confusion.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path)
    main(parser.parse_args().path)
