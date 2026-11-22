"""F11 (#69): drain, recoil, Substitute and end-of-turn HP changes in the payoff table."""

import warnings

import pytest
from poke_env.battle import Effect, Move, Status, Weather
from test_damage import make_battle

from pokeagent.config import PAYOFF_KO_BONUS
from pokeagent.decision._payoff_state import move_effect, residual, scenario_copy, substitute_hp
from pokeagent.decision._turn import TurnEffect, turn_value
from pokeagent.decision.damage import random_battle_stats
from pokeagent.decision.payoff import PayoffApproximationWarning, build_payoff_table
from pokeagent.interfaces import SetBelief

ME, OPP = "p1: Attacker", "p2: Defender"
NONE = TurnEffect()


# --- The turn model ----------------------------------------------------------------------


def test_drain_heals_part_of_the_damage_dealt_up_to_full_hp():
    giga = TurnEffect((0.4, 0.4), drain=0.5)
    assert turn_value(0.5, 1, giga, NONE, 1) == pytest.approx(0.4 + 0.2)
    assert turn_value(1.0, 1, giga, NONE, 1) == pytest.approx(0.4)  # already full
    # Drain counts only damage actually dealt: 0.4 into a target with 0.1 HP left heals 0.05.
    assert turn_value(0.5, 0.1, giga, NONE, 1, ko_bonus=0) == pytest.approx(0.1 + 0.05)


def test_recoil_costs_hp_and_can_knock_out_the_user():
    bird = TurnEffect((0.6, 0.6), recoil=1 / 3)
    assert turn_value(1, 1, bird, NONE, 1) == pytest.approx(0.6 - 0.2)
    assert turn_value(0.1, 1, bird, NONE, 1) == pytest.approx(0.6 - 0.1 - PAYOFF_KO_BONUS)


def test_end_of_turn_changes_apply_only_to_pokemon_still_standing():
    assert turn_value(0.5, 1, NONE, NONE, 1, residuals=(1 / 16, 0)) == pytest.approx(1 / 16)
    assert turn_value(1.0, 1, NONE, NONE, 1, residuals=(1 / 16, 0)) == 0  # Leftovers at full
    assert turn_value(1, 1, NONE, NONE, 1, residuals=(0, -1 / 16)) == pytest.approx(1 / 16)
    # Burn finishing them off earns the KO bonus; a fainted Pokemon takes no burn damage.
    assert turn_value(1, 0.05, NONE, NONE, 1, residuals=(0, -1 / 16)) == pytest.approx(
        0.05 + PAYOFF_KO_BONUS
    )
    ko = TurnEffect((1.0, 1.0))
    assert turn_value(1, 0.5, ko, NONE, 1, residuals=(0, -1 / 16)) == pytest.approx(
        0.5 + PAYOFF_KO_BONUS
    )


def test_substitute_costs_a_quarter_and_absorbs_later_hits():
    sub, hit = TurnEffect(substitute=True, blocked_by_protect=False), TurnEffect((0.3, 0.3))
    assert turn_value(1, 1, sub, hit, 1) == pytest.approx(-0.25)  # up first, takes the hit
    assert turn_value(1, 1, sub, hit, 0) == pytest.approx(-0.3 - 0.25)  # hit first
    assert turn_value(0.25, 1, sub, NONE, 1) == 0  # fails at 1/4 HP or below
    # Their existing Substitute absorbs our 0.4; at 0.1 HP left it absorbs only that.
    attack = TurnEffect((0.4, 0.4))
    assert turn_value(1, 1, attack, NONE, 1, substitutes=(0, 0.25)) == 0
    # A status aimed at a Pokemon behind a Substitute does nothing.
    para = TurnEffect(future=0.2, future_needs_target=True)
    assert turn_value(1, 1, para, NONE, 1, substitutes=(0, 0.25)) == 0
    sound = TurnEffect((0.4, 0.4), blocked_by_substitute=False)
    assert turn_value(1, 1, sound, NONE, 1, substitutes=(0, 0.25)) == pytest.approx(0.4)


# --- From live battle objects ------------------------------------------------------------


def effect(move, attacker="Venusaur", defender="Blissey"):
    battle = scenario_copy(make_battle(attacker, defender, (move,)), ME, OPP)  # fills stats
    return move_effect(battle, ME, OPP, Move(move, gen=9))


def test_drain_and_recoil_are_converted_to_the_users_hp():
    def hp(species):
        return random_battle_stats(species, 100)["hp"]

    assert effect("gigadrain").drain == pytest.approx(0.5 * hp("blissey") / hp("venusaur"))
    assert effect("bravebird", "Staraptor").recoil == pytest.approx(
        0.33 * hp("blissey") / hp("staraptor")
    )
    assert not effect("boomburst", "Exploud").blocked_by_substitute  # sound
    assert not effect("stealthrock").blocked_by_substitute  # hits the side
    assert effect("thunderwave").blocked_by_substitute
    assert not effect("swordsdance").blocked_by_substitute and effect("substitute").substitute


def test_residual_from_item_status_and_weather():
    battle = make_battle("Garchomp", "Blissey")
    mon = battle.get_pokemon(ME)
    assert residual(mon, battle) == 0
    mon.item = "leftovers"
    assert residual(mon, battle) == pytest.approx(1 / 16)
    mon.item = "blacksludge"  # Garchomp is not Poison type
    assert residual(mon, battle) == pytest.approx(-1 / 8)
    mon.item, mon._status, mon._status_counter = None, Status.TOX, 2
    assert residual(mon, battle) == pytest.approx(-3 / 16)
    blissey = battle.get_pokemon(OPP)
    battle._weather = {Weather.SANDSTORM: 1}
    assert residual(blissey, battle) == pytest.approx(-1 / 16)
    assert residual(mon, battle) == pytest.approx(-3 / 16)  # Ground: no sand damage
    blissey._ability = "magicguard"
    assert residual(blissey, battle) == 0
    assert residual(None, battle) == 0


# --- In the payoff table -----------------------------------------------------------------


def rows(battle, beliefs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PayoffApproximationWarning)
        table = build_payoff_table(battle, beliefs, None)
    return {a.name: row for a, row in zip(table.our_actions, table.values, strict=True)}


def test_their_substitute_soaks_our_attack():
    battle = make_battle("Garchomp", "Blissey", ("earthquake",))
    battle._team_size = {"p1": 1, "p2": 1}
    beliefs = {"blissey": SetBelief("blissey", move_probs={"seismictoss": 1})}
    before = rows(battle, beliefs)["earthquake"][0]
    battle.get_pokemon(OPP)._effects[Effect.SUBSTITUTE] = 0
    assert substitute_hp(battle.get_pokemon(OPP)) == 0.25
    assert rows(battle, beliefs)["earthquake"][0] < before - 0.1
