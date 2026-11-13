"""Does poke-env's damage calculator work when the opponent's details are hidden? (check 5)

Role F's payoff table needs "how much damage does each of our moves do to their active
Pokemon?". poke-env has a damage calculator, but it needs the defender's stats, item and
ability, which we usually don't know for the opponent.

This spike plays real games with a probe bot. Every turn, for each of our damaging moves, it
computes the damage range two ways:

  public : what we could know in a real game. The opponent's stats are filled in from species +
           level using the Random Battle generator's fixed spreads (85 EVs and 31 IVs in every
           stat, neutral nature); item and ability are whatever has been revealed so far.
  oracle : the same calculation with the opponent's TRUE stats, item and ability, read from the
           other bot's battle object. Only possible in an experiment; it isolates how much error
           comes from hidden information rather than from the calculator itself.

It then plays the move with the highest predicted damage and, after the game, reads the real
damage from the log. We report how often the real damage fell inside each predicted range.

Usage (local server running):
    python scripts/spikes/damage_calc_check.py --games 100
"""

import argparse
import asyncio
import math
from collections import Counter

from poke_env import AccountConfiguration
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.data import GenData
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.stats import compute_raw_stats

from pokeagent.config import BATTLE_FORMAT

DATA = GenData.from_gen(9)
STAT_KEYS = ["hp", "atk", "def", "spa", "spd", "spe"]


def random_battle_stats(species: str, level: int) -> dict:
    """Stats a Random Battle Pokemon has if its spread is the default (85 EVs, 31 IVs, neutral).

    The generator sometimes lowers HP EVs slightly, and sets Attack to 0 EVs/IVs for sets with no
    physical moves and Speed to 0 for Trick Room / Gyro Ball sets; we ignore those here.
    """
    raw = compute_raw_stats(species, [85] * 6, [31] * 6, level, "serious", DATA)
    return dict(zip(STAT_KEYS, raw, strict=True))


def key_of(team: dict, mon) -> str:
    return next(k for k, v in team.items() if v is mon)


class ProbePlayer(Player):
    peer: Player = None  # the opponent bot, peeked at only to compute the oracle

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[dict] = []
        self.errors: Counter = Counter()

    def choose_move(self, battle):
        # An exception inside choose_move stalls the game forever, so never let one escape.
        try:
            return self._probe(battle)
        except Exception as e:  # noqa: BLE001
            self.errors[f"probe crashed: {type(e).__name__}: {e}"[:120]] += 1
            return self.choose_random_move(battle)

    def _probe(self, battle):
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        if battle.force_switch or me is None or opp is None or not battle.available_moves:
            return self.choose_random_move(battle)
        me_key, opp_key = key_of(battle.team, me), key_of(battle.opponent_team, opp)

        # What happens if we call the calculator with nothing filled in?
        if opp.stats.get("atk") is None:
            try:
                calculate_damage(me_key, opp_key, battle.available_moves[0], battle)
                self.errors["works_without_opponent_stats"] += 1
            except AssertionError as e:
                self.errors[f"AssertionError: {e}"] += 1
            except Exception as e:  # noqa: BLE001 (spike: record anything)
                self.errors[f"{type(e).__name__}"] += 1
            opp.stats = random_battle_stats(opp.species, opp.level)

        peer_battle = self.peer.battles[battle.battle_tag]
        truth = peer_battle.team.get(opp_key)

        best, best_dmg, preds = None, -1.0, []
        for move in battle.available_moves:
            if move.base_power == 0 and move.category.name == "STATUS":
                continue
            try:
                lo, hi = calculate_damage(me_key, opp_key, move, battle)
            except Exception as e:  # noqa: BLE001
                self.errors[f"calc failed on {move.id}: {type(e).__name__}"] += 1
                continue
            pub_max_hp = opp.stats["hp"]
            oracle = None
            if truth is not None:
                saved = (opp.stats, opp._item, opp._ability)
                opp.stats, opp._item, opp._ability = truth.stats, truth.item, truth.ability
                try:
                    olo, ohi = calculate_damage(me_key, opp_key, move, battle)
                    oracle = (100 * olo / truth.max_hp, 100 * ohi / truth.max_hp)
                except Exception:  # noqa: BLE001
                    oracle = None
                opp.stats, opp._item, opp._ability = saved
            pred = {
                "move": move.id,
                "public": (100 * lo / pub_max_hp, 100 * hi / pub_max_hp),
                "oracle": oracle,
                "item_known": opp.item not in (None, "unknown_item"),
            }
            preds.append(pred)
            if hi > best_dmg:
                best, best_dmg = move, hi
        if best is None:
            return self.choose_random_move(battle)
        self.records.append(
            {
                "battle": battle.battle_tag,
                "turn": battle.turn,
                "me_role": battle.player_role,
                "opp_species": opp.species,
                "opp_hp_pct": round(100 * opp.current_hp_fraction),
                "chosen": next(p for p in preds if p["move"] == best.id),
            }
        )
        return self.create_order(best)


def to_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def observed_damage(stream: list[list[str]], turn: int, me: str, move_id: str, hp_before: int):
    """Real damage % our move did this turn, or a reason to skip the turn."""
    opp = "p2" if me == "p1" else "p1"
    lines = []
    in_turn = False
    for m in stream:
        if len(m) > 2 and m[1] == "turn":
            if in_turn:
                break
            in_turn = int(m[2]) == turn
            continue
        if in_turn:
            lines.append(m)
    hp = hp_before
    for i, m in enumerate(lines):
        if len(m) < 3:
            continue
        if m[1] in ("switch", "drag") and m[2].startswith(opp):
            return "opponent_switched"
        if m[1] in ("-damage", "-heal", "-sethp") and m[2].startswith(opp) and len(m) > 3:
            hp = 0 if m[3].startswith("0") else int(m[3].split("/")[0])
        if m[1] == "move" and m[2].startswith(me) and to_id(m[3]) == move_id:
            if any(x in ("[miss]", "[still]", "[notarget]") for x in m):
                return "missed"
            after = lines[i + 1 :]
            hits = []
            for a in after:
                if len(a) > 1 and a[1] == "move":
                    break
                if len(a) > 1 and a[1] == "-crit":
                    return "crit"
                if len(a) > 1 and a[1] == "-activate" and "Substitute" in "|".join(a):
                    return "substitute"
                if len(a) > 1 and a[1] in ("-immune", "-fail"):
                    return "immune_or_failed"
                if a[1:2] == ["-damage"] and a[2].startswith(opp) and len(a) == 4:
                    hits.append(0 if a[3].startswith("0") else int(a[3].split("/")[0]))
            if len(hits) != 1:
                return "no_hit" if not hits else "multi_hit"
            new_hp = hits[0]
            return {"dealt": hp - new_hp, "ko": new_hp == 0, "hp_before": hp}
    return "move_not_found"


def in_range(dealt: int, ko: bool, hp_before: int, rng) -> bool:
    lo, hi = rng
    if ko:  # damage is capped at remaining HP: consistent if the max roll could KO
        return hi >= hp_before - 1
    # the log shows rounded-up percentages, so allow one point either side
    return math.floor(lo) - 1 <= dealt <= math.ceil(hi) + 1


async def main(games: int) -> None:
    probe = ProbePlayer(
        account_configuration=AccountConfiguration("calcprobe", None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
    )
    peer = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration("calcpeer", None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
    )
    probe.peer = peer
    await probe.battle_against(peer, n_battles=games)

    outcomes: Counter = Counter()
    hits = Counter()
    for r in probe.records:
        b = probe.battles[r["battle"]]
        obs = observed_damage(
            b._replay_data, r["turn"], r["me_role"], r["chosen"]["move"], r["opp_hp_pct"]
        )
        if isinstance(obs, str):
            outcomes[obs] += 1
            continue
        outcomes["measured"] += 1
        known = "item revealed" if r["chosen"]["item_known"] else "item hidden"
        hits[f"public in range ({known})"] += in_range(
            obs["dealt"], obs["ko"], obs["hp_before"], r["chosen"]["public"]
        )
        hits[f"n ({known})"] += 1
        if r["chosen"]["oracle"]:
            hits["oracle in range"] += in_range(
                obs["dealt"], obs["ko"], obs["hp_before"], r["chosen"]["oracle"]
            )
            hits["n oracle"] += 1
            if not in_range(obs["dealt"], obs["ko"], obs["hp_before"], r["chosen"]["oracle"]):
                outcomes[f"oracle miss example: {r['chosen']['move']} vs {r['opp_species']}"] += 1

    print(f"{games} games, {len(probe.records)} probe decisions, probe won {probe.n_won_battles}")
    print("\nCalling the calculator before filling in opponent stats:")
    for k, v in probe.errors.most_common(8):
        print(f"  {v:>5}  {k}")
    print("\nTurns we could measure, and why the rest were skipped:")
    for k, v in outcomes.most_common():
        if "example" not in k:
            print(f"  {v:>5}  {k}")
    print("\nReal damage inside the predicted range:")
    for label in ("item hidden", "item revealed"):
        n = hits[f"n ({label})"]
        if n:
            ok = hits[f"public in range ({label})"]
            print(f"  public, {label:<14} {ok:>5}/{n:<5} ({100 * ok / n:.1f}%)")
    if hits["n oracle"]:
        ok, n = hits["oracle in range"], hits["n oracle"]
        print(f"  oracle (true hidden info) {ok:>5}/{n:<5} ({100 * ok / n:.1f}%)")
    misses = [(k, v) for k, v in outcomes.most_common() if "example" in k][:10]
    if misses:
        print("\nOracle misses (calculator itself wrong):")
        for k, v in misses:
            print(f"  {v:>3}  {k.split(': ', 1)[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=100)
    asyncio.run(main(parser.parse_args().games))
