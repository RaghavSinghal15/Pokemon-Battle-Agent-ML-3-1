"""Regression examples for the four failure groups found in domain review of F4."""

import math

import pytest
from poke_env.battle import Move, Pokemon, Status
from test_payoff import add_reserve, build, table_battle

from pokeagent.decision._payoff_state import move_effect, scenario_copy
from pokeagent.decision._turn import TurnEffect, turn_value
from pokeagent.decision.payoff import PayoffUnavailable, build_payoff_table


def test_full_attack_pool_drops_status_and_switch_mass_without_changing_forecast():
    battle, beliefs = table_battle(candidates={"seismictoss": 0.75, "nightshade": 0.25})
    forecast = {"attack": 1 / 3, "non_damaging": 1 / 3, "switch": 1 / 3}
    assert build(battle, beliefs, forecast).opp_probs == pytest.approx([0.25, 0.75])
    assert forecast == dict.fromkeys(forecast, 1 / 3)


def test_revealed_moves_alone_do_not_prove_a_class_impossible():
    battle, _ = table_battle()
    battle.opponent_active_pokemon.moves["seismictoss"] = Move("seismictoss", gen=9)
    with pytest.raises(PayoffUnavailable, match="non_damaging.*no move candidates"):
        build_payoff_table(battle, {}, {"attack": 0.5, "non_damaging": 0.5, "switch": 0})


def test_no_remaining_forecast_mass_is_an_explicit_error():
    battle, beliefs = table_battle()
    with pytest.raises(PayoffUnavailable, match="no mass"):
        build_payoff_table(battle, beliefs, {"attack": 0, "non_damaging": 0, "switch": 1})


@pytest.mark.parametrize("move", ["meteorbeam", "electroshot", "solarbeam"])
def test_charge_candidates_do_not_break_other_columns(move):
    battle, beliefs = table_battle(candidates={move: 1, "seismictoss": 1})
    table = build(battle, beliefs)
    assert len(table.opp_actions) == 2
    assert all(math.isfinite(value) for value in table.values[0])
    scenario = scenario_copy(battle, "p1: Attacker", "p2: Defender")
    effect = move_effect(scenario, "p2: Defender", "p1: Attacker", Move(move, gen=9))
    assert effect.damage == (0, 0)


@pytest.mark.parametrize(
    "move", ["kingsshield", "spikyshield", "banefulbunker", "silktrap", "burningbulwark"]
)
def test_protect_variants_block_damage_on_either_side(move):
    battle, beliefs = table_battle(moves=(move,))
    assert build(battle, beliefs).values[0][0] == 0
    battle, beliefs = table_battle(candidates={move: 1})
    assert build(battle, beliefs).values[0][0] == 0


@pytest.mark.parametrize("move", ["explosion", "selfdestruct", "mistyexplosion"])
def test_self_ko_attack_adapter_and_unknown_switch_cost(move):
    battle, beliefs = table_battle(moves=(move,), candidates={move: 1})
    assert math.isfinite(build(battle, beliefs).values[0][0])
    scenario = scenario_copy(battle, "p1: Attacker", "p2: Defender")
    effect = move_effect(scenario, "p1: Attacker", None, Move(move, gen=9))
    assert effect.self_ko
    assert turn_value(0.6, 1, effect, TurnEffect(), 1) == pytest.approx(-0.85)


def test_self_ko_happens_on_miss_or_protect_but_not_when_unable_to_act():
    explosion = TurnEffect((0.4, 0.4), hit_chance=0, self_ko=True)
    assert turn_value(1, 1, explosion, TurnEffect(), 1) == pytest.approx(-1.25)
    explosion = TurnEffect((0.4, 0.4), self_ko=True)
    assert turn_value(1, 1, explosion, TurnEffect(protect=True), 0) == pytest.approx(-1.25)
    sleeping = TurnEffect((0.4, 0.4), self_ko=True, act_chance=0)
    assert turn_value(1, 1, sleeping, TurnEffect(), 1) == 0


def test_earlier_ko_cancels_self_ko_attempt():
    explosion = TurnEffect((1, 1), self_ko=True)
    assert turn_value(0.2, 1, explosion, TurnEffect((0.3, 0.3)), 0) == pytest.approx(-0.45)


@pytest.mark.parametrize("status,chance", [(Status.SLP, 1 / 3), (Status.FRZ, 0.2)])
def test_status_action_probability_is_separate_from_accuracy(status, chance):
    battle, _ = table_battle()
    battle.active_pokemon.status = status
    scenario = scenario_copy(battle, "p1: Attacker", "p2: Defender")
    effect = move_effect(scenario, "p1: Attacker", "p2: Defender", Move("seismictoss", gen=9))
    assert effect.act_chance == chance
    assert effect.hit_chance == 1
    assert battle.active_pokemon.status == status
    effect = TurnEffect((0.4, 0.4), act_chance=chance)
    assert turn_value(1, 1, effect, TurnEffect((0.1, 0.1)), 1) == pytest.approx(0.4 * chance - 0.1)


def test_phantom_reserve_is_skipped_and_does_not_hide_an_unseen_member():
    battle, beliefs = table_battle()
    phantom = Pokemon(gen=9, species="zoroark")  # The disguise object left at 0/0 HP.
    battle.opponent_team["p2: Disguise"] = phantom
    assert phantom.max_hp == 0 and not phantom.fainted
    forecast = {"attack": 0.5, "non_damaging": 0, "switch": 0.5}
    assert build(battle, beliefs, forecast).opp_probs == [1.0]
    battle._team_size["p2"] = 2
    assert build(battle, beliefs, forecast).opp_probs == [0.5, 0.5]


def test_zero_hp_non_fainted_reserve_is_not_scored_as_alive():
    battle, beliefs = table_battle()
    reserve = add_reserve(battle, "p2")
    reserve._current_hp = 0
    battle._team_size["p2"] = 2
    assert not reserve.fainted
    forecast = {"attack": 0.5, "non_damaging": 0, "switch": 0.5}
    assert build(battle, beliefs, forecast).opp_probs == [1.0]
