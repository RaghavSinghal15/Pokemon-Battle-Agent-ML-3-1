"""Lasting effects in the payoff table (F8 #66, F9 #67, F10 #68): stat changes, status and
hazards are valued by how much they change next turn's table."""

import warnings

import pytest
from poke_env.battle import Move, SideCondition, Status
from test_damage import make_battle

from pokeagent.decision import payoff
from pokeagent.decision._payoff_state import lasting_effect
from pokeagent.decision._turn import TurnEffect, turn_value
from pokeagent.decision.payoff import PayoffApproximationWarning, build_payoff_table
from pokeagent.interfaces import SetBelief

ME, OPP = "p1: Attacker", "p2: Defender"


def effect_of(battle, move, user=ME, target=OPP):
    return lasting_effect(battle, user, target, Move(move, gen=9))


# --- What a move leaves behind --------------------------------------------------------


def test_stat_changes_on_user_and_target():
    battle = make_battle("Garchomp", "Blissey")
    dance = effect_of(battle, "swordsdance")
    assert dance.battle.get_pokemon(ME).boosts["atk"] == 2 and dance.needs_user
    assert battle.get_pokemon(ME).boosts["atk"] == 0  # the real battle is not touched
    assert effect_of(battle, "dracometeor").battle.get_pokemon(ME).boosts["spa"] == -2
    wind = effect_of(battle, "icywind")
    assert wind.battle.get_pokemon(OPP).boosts["spe"] == -1 and wind.needs_target
    battle.get_pokemon(ME)._boosts["atk"] = 6
    assert effect_of(battle, "swordsdance") is None  # already maxed: nothing changes


def test_status_respects_immunities():
    battle = make_battle("Jolteon", "Blissey")
    assert effect_of(battle, "thunderwave").battle.get_pokemon(OPP).status == Status.PAR
    assert effect_of(make_battle("Jolteon", "Garchomp"), "thunderwave") is None  # Ground
    assert effect_of(make_battle("Jolteon", "Venusaur"), "spore") is None  # Grass vs powder
    assert effect_of(make_battle("Jolteon", "Corviknight"), "toxic") is None  # Steel
    battle.get_pokemon(OPP)._status = Status.BRN
    assert effect_of(battle, "thunderwave") is None  # already has a status


def test_hazards_up_to_their_maximum_and_removal():
    battle = make_battle("Garchomp", "Blissey")
    rocks = effect_of(battle, "stealthrock")
    assert rocks.battle.opponent_side_conditions[SideCondition.STEALTH_ROCK] == 1
    assert not rocks.needs_user and not rocks.needs_target
    battle._opponent_side_conditions[SideCondition.STEALTH_ROCK] = 1
    assert effect_of(battle, "stealthrock") is None
    # Nothing on our side to remove, but Rapid Spin still raises the user's Speed by 1.
    spin = effect_of(battle, "rapidspin")
    assert spin.battle.get_pokemon(ME).boosts["spe"] == 1 and spin.needs_user
    battle._side_conditions[SideCondition.SPIKES] = 2
    spin = effect_of(battle, "rapidspin")
    assert SideCondition.SPIKES not in spin.battle.side_conditions
    assert spin.battle.opponent_side_conditions[SideCondition.STEALTH_ROCK] == 1


# --- When the value counts -------------------------------------------------------------


def test_future_counts_only_when_the_move_lands():
    setup = TurnEffect(future=0.3, future_needs_user=True)
    assert turn_value(1, 1, setup, TurnEffect(), 1) == pytest.approx(0.3)
    assert turn_value(1, 1, TurnEffect(future=0.3, hit_chance=0.5), TurnEffect(), 1) == (
        pytest.approx(0.15)
    )
    # Knocked out after setting up (they move second and hit for 100%): the boost is lost.
    ko = TurnEffect((1.0, 1.0))
    assert turn_value(1, 1, setup, ko, 1) == pytest.approx(-1 - 0.25)
    # Protect blocks an effect aimed at the target, not one the user gives itself.
    para = TurnEffect(future=0.2, future_needs_target=True)
    assert turn_value(1, 1, para, TurnEffect(protect=True), 0) == 0
    dance = TurnEffect(future=0.2, blocked_by_protect=False, future_needs_user=True)
    assert turn_value(1, 1, dance, TurnEffect(protect=True), 0) == pytest.approx(0.2)


# --- In the payoff table ---------------------------------------------------------------


def table(battle, beliefs, effect_turns, monkeypatch):
    monkeypatch.setattr(payoff, "PAYOFF_EFFECT_TURNS", effect_turns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PayoffApproximationWarning)
        result = build_payoff_table(battle, beliefs, None)
    return {a.name: row for a, row in zip(result.our_actions, result.values, strict=True)}


def setup_battle(our_hp="100/100"):
    battle = make_battle("Garchomp", "Blissey", ("earthquake", "swordsdance"))
    battle.active_pokemon.set_hp(our_hp)
    battle._team_size = {"p1": 1, "p2": 1}
    return battle, {"blissey": SetBelief("blissey", move_probs={"seismictoss": 1})}


def test_swords_dance_gains_value_when_it_is_safe(monkeypatch):
    battle, beliefs = setup_battle()
    without = table(battle, beliefs, 0.0, monkeypatch)
    with_effects = table(battle, beliefs, 1.0, monkeypatch)
    assert with_effects["earthquake"] == without["earthquake"]  # no lasting effect
    assert with_effects["swordsdance"][0] > without["swordsdance"][0] + 0.1


def test_swords_dance_is_worthless_if_we_are_knocked_out_this_turn(monkeypatch):
    # Seismic Toss does 100 HP; at 60 HP Garchomp sets up first, then faints.
    battle, beliefs = setup_battle("60/357")
    assert table(battle, beliefs, 1.0, monkeypatch)["swordsdance"] == pytest.approx(
        table(battle, beliefs, 0.0, monkeypatch)["swordsdance"]
    )


def test_their_status_move_counts_against_us(monkeypatch):
    battle, _ = setup_battle()
    beliefs = {"blissey": SetBelief("blissey", move_probs={"willowisp": 1})}
    without = table(battle, beliefs, 0.0, monkeypatch)
    with_effects = table(battle, beliefs, 1.0, monkeypatch)
    # A burn halves Garchomp's Earthquake next turn, so their Will-O-Wisp costs us value.
    assert with_effects["earthquake"][0] < without["earthquake"][0] - 0.05


def test_lookahead_works_with_reserves_on_both_sides(monkeypatch):
    """Live battles have reserves; next turn's table must still find our switch rows."""
    from test_payoff import add_reserve

    battle, beliefs = setup_battle()
    battle._team_size = {"p1": 2, "p2": 2}
    battle._available_switches = [add_reserve(battle, "p1")]
    add_reserve(battle, "p2")
    rows = table(battle, beliefs, 1.0, monkeypatch)
    assert set(rows) == {"earthquake", "swordsdance", "snorlax"}
    assert rows["swordsdance"][0] > table(battle, beliefs, 0.0, monkeypatch)["swordsdance"][0]


def test_hazard_value_counts_every_remaining_switch_in():
    from test_payoff import add_reserve

    from pokeagent.decision._payoff_state import hazard_value

    battle = make_battle("Garchomp", "Blissey", ("stealthrock", "earthquake"))
    add_reserve(battle, "p2")  # a Snorlax: Stealth Rock costs it 1/8
    rocks = effect_of(battle, "stealthrock")
    # One known reserve plus four unseen ones, each 1/8 (neutral): 5/8 of a Pokemon's HP.
    assert hazard_value(battle, rocks.battle, unseen_opponents=4) == pytest.approx(5 / 8, abs=0.01)
    # Removing two layers of Spikes from our side saves our reserve 1/6.
    battle._side_conditions[SideCondition.SPIKES] = 2
    add_reserve(battle, "p1")
    spin = effect_of(battle, "rapidspin")
    assert hazard_value(battle, spin.battle, unseen_opponents=0) == pytest.approx(1 / 6, abs=0.01)


def test_stealth_rock_row_gains_the_weighted_hazard_value(monkeypatch):
    from test_payoff import add_reserve

    battle = make_battle("Garchomp", "Blissey", ("stealthrock", "earthquake"))
    battle._team_size = {"p1": 1, "p2": 6}
    add_reserve(battle, "p2")
    beliefs = {"blissey": SetBelief("blissey", move_probs={"seismictoss": 1})}
    monkeypatch.setattr(payoff, "PAYOFF_HAZARD_WEIGHT", 0.5)
    on = table(battle, beliefs, 1.0, monkeypatch)
    monkeypatch.setattr(payoff, "PAYOFF_HAZARD_WEIGHT", 0.0)
    off = table(battle, beliefs, 1.0, monkeypatch)
    # Against Seismic Toss, Garchomp sets the rock in every cell: +0.5 x 5/8.
    assert on["stealthrock"][0] - off["stealthrock"][0] == pytest.approx(0.5 * 5 / 8, abs=0.01)
    assert on["earthquake"] == off["earthquake"]


def test_zero_weights_skip_the_lookahead(monkeypatch):
    battle, beliefs = setup_battle()
    calls = []
    monkeypatch.setattr(payoff, "lasting_effect", lambda *a: calls.append(a))
    monkeypatch.setattr(payoff, "PAYOFF_HAZARD_WEIGHT", 0.0)
    table(battle, beliefs, 0.0, monkeypatch)
    assert calls == []  # both weights zero: no lasting effect is even computed
