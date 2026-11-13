"""Compare the replay view and the live view of the same battles (foundation checks 2 and 3).

For every battle, every perspective (p1 and p2) and every turn, we build three snapshots:

  R  replay   : the spectator log, re-parsed by poke-env (what training data will look like)
  L  live     : the live player's own battle object at the moment it chose a move
  S  stream   : the live player's received messages, re-parsed the same way as R

and count how often each field disagrees. R vs L tells us whether "train on replays, play live"
sees the same world. R vs S tells us whether a live converter can simply re-parse the player's
own message stream with the replay parser (one code path for both).

Usage:
    python scripts/spikes/compare_views.py data/interim/local_games.jsonl
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from replay_parse import parse_log  # noqa: E402
from snapshot import battle_summary  # noqa: E402

MON_FIELDS_PUBLIC = ("hp", "status", "fainted", "active", "boosts", "tera_type", "terastallized")
MON_FIELDS_REVEALED = ("moves", "item", "ability")
BATTLE_FIELDS = ("my_side_conditions", "opp_side_conditions", "weather", "fields", "opp_used_tera")


def public_hp(hp_raw: str) -> float:
    """What a spectator sees for an exact HP value.

    Showdown's rule (sim/pokemon.ts, getHealth): ceil(100 * hp / max) percent, except that a
    Pokemon that is not at full HP never shows 100 (it shows 99).
    """
    cur, mx = (int(x) for x in hp_raw.split("/"))
    if cur == 0 or mx == 0:
        return 0.0
    pct = math.ceil(100 * cur / mx)
    if pct == 100 and cur < mx:
        pct = 99
    return round(pct / 100, 2)


def snapshots_from_log(battle_id, lines, username):
    snaps = {}

    def grab(battle):
        snaps.setdefault(battle.turn, battle_summary(battle))

    parse_log(battle_id, lines, username, on_turn=grab)
    return snaps


def diff(a: dict, b: dict, counts: Counter, examples: dict, tag: str, own_side_is_live: bool):
    """Count field disagreements between snapshot a (replay) and b (other view)."""
    for field in BATTLE_FIELDS:
        counts[f"{tag}:{field}:checked"] += 1
        if a[field] != b[field]:
            counts[f"{tag}:{field}:diff"] += 1
            examples.setdefault(f"{tag}:{field}", (a["turn"], a[field], b[field]))
    for side in ("me", "opp"):
        for species, ma in a[side].items():
            counts[f"{tag}:{side}:species:checked"] += 1
            mb = b[side].get(species)
            if mb is None:
                counts[f"{tag}:{side}:species:diff"] += 1
                examples.setdefault(f"{tag}:{side}:species", (a["turn"], species, None))
                continue
            fields = MON_FIELDS_PUBLIC
            if not (side == "me" and own_side_is_live):
                fields = fields + MON_FIELDS_REVEALED
            for field in fields:
                va, vb = ma[field], mb[field]
                if field == "hp" and side == "me" and own_side_is_live:
                    vb = public_hp(mb["hp_raw"])  # convert exact own HP to the public view
                counts[f"{tag}:{side}:{field}:checked"] += 1
                if va != vb:
                    counts[f"{tag}:{side}:{field}:diff"] += 1
                    examples.setdefault(f"{tag}:{side}:{field}", (a["turn"], species, va, vb))


def main(path: Path) -> None:
    counts: Counter = Counter()
    examples: dict = {}
    n_battles = n_turns = 0
    for line in path.open():
        rec = json.loads(line)
        n_battles += 1
        for role, username in rec["players"].items():
            replay = snapshots_from_log(rec["battle_id"], rec["spectator_log"], username)
            stream = snapshots_from_log(rec["battle_id"], rec["streams"][role], username)
            live = {int(t): s for t, s in rec["live"][role].items()}
            for turn, r in replay.items():
                n_turns += 1
                if turn in live:
                    diff(r, live[turn], counts, examples, "R_vs_L", own_side_is_live=True)
                else:
                    counts["R_vs_L:turn_missing_in_live"] += 1
                if turn in stream:
                    # The stream has exact own HP, so convert it like a live state.
                    s = json.loads(json.dumps(stream[turn]))
                    for mon in s["me"].values():
                        mon["hp"] = public_hp(mon["hp_raw"])
                    diff(r, s, counts, examples, "R_vs_S", own_side_is_live=False)
                else:
                    counts["R_vs_S:turn_missing_in_stream"] += 1

    print(f"{n_battles} battles, {n_turns} (perspective, turn) decision points\n")
    for tag in ("R_vs_L", "R_vs_S"):
        print(f"== {tag}")
        keys = sorted(
            {k.rsplit(":", 1)[0] for k in counts if k.startswith(tag) and k.count(":") >= 2}
        )
        for k in keys:
            checked, d = counts[f"{k}:checked"], counts[f"{k}:diff"]
            if checked:
                flag = "" if d == 0 else "   <-- differs"
                name = k[len(tag) + 1 :]
                pct = 100 * d / checked
                print(f"  {name:<28} {d:>6} / {checked:<6} ({pct:5.2f}%){flag}")
        for k, v in counts.items():
            if k.startswith(tag) and "missing" in k:
                print(f"  {k}: {v}")
        print()
    print("== first example of each disagreement (turn, species/value, replay, other)")
    for k, v in sorted(examples.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path)
    main(parser.parse_args().path)
