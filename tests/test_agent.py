"""Legal action mapping and explicit fallback behavior of the F5 player."""

import json

import pytest
from poke_env.battle import Pokemon
from test_damage import make_battle
from test_payoff import add_reserve

from pokeagent.agent import player
from pokeagent.agent.player import ExplainableAgent, flat_beliefs
from pokeagent.beliefs import tracker
from pokeagent.config import BATTLE_FORMAT
from pokeagent.decision.payoff import PayoffUnavailable
from pokeagent.interfaces import Action, PayoffTable, SetBelief


@pytest.fixture(autouse=True)
def public_sets(tmp_path, monkeypatch):
    """Small public set definitions; fast agent tests need no Showdown checkout."""
    path = tmp_path / "agent-sets.json"
    definitions = {}
    for species, moves in (
        ("blissey", ["Seismic Toss", "Protect"]),
        ("pikachu", ["Surf", "Thunderbolt"]),
    ):
        definitions[species] = {
            "sets": [
                {
                    "role": "Test support",
                    "movepool": moves,
                    "abilities": ["Static"],
                    "teraTypes": ["Normal"],
                }
            ]
        }
    path.write_text(json.dumps(definitions), encoding="utf-8")
    monkeypatch.setattr(player, "RANDOM_BATTLE_SETS_PATH", path)
    monkeypatch.setattr(tracker, "RANDOM_BATTLE_SETS_PATH", path)


@pytest.fixture
def agent():
    return ExplainableAgent(battle_format=BATTLE_FORMAT, start_listening=False)


@pytest.mark.parametrize("strategy", ["best_response", "lp"])
def test_table_selects_legal_damage_move(strategy, monkeypatch):
    battle = make_battle("Blissey", "Blissey", ("splash", "seismictoss"))
    battle._team_size = {"p1": 1, "p2": 1}
    monkeypatch.setattr(
        player,
        "flat_beliefs",
        lambda b: {"blissey": SetBelief("blissey", move_probs={"splash": 1})},
    )
    bot = ExplainableAgent(battle_format=BATTLE_FORMAT, strategy=strategy, start_listening=False)
    order = bot.choose_move(battle)
    assert order.order.id == "seismictoss"
    assert bot.last_decision.strategy == strategy
    assert bot.decision_counts[strategy] == 1
    assert not bot.fallback_reasons


def test_switch_row_maps_to_actual_legal_pokemon(agent, monkeypatch):
    battle = make_battle()
    reserve = add_reserve(battle, "p1")
    battle._available_switches = [reserve]
    table = PayoffTable([Action("switch", reserve.species)], [Action("move", "splash")], [[1]], [1])
    monkeypatch.setattr(player, "build_payoff_table", lambda *args: table)
    assert agent.choose_move(battle).order is reserve


def test_missing_table_records_reason_and_uses_greedy_damage(agent, monkeypatch):
    battle = make_battle(moves=("splash", "seismictoss"))

    def unavailable(*args):
        raise PayoffUnavailable("test missing information")

    monkeypatch.setattr(player, "build_payoff_table", unavailable)
    assert agent.choose_move(battle).order.id == "seismictoss"
    assert agent.last_decision.strategy == "baseline"
    assert "test missing information" in agent.last_decision.explanation
    assert sum(agent.fallback_reasons.values()) == 1


def test_forced_switch_uses_legal_reserve_without_building_table(agent, monkeypatch):
    battle = make_battle()
    reserve = add_reserve(battle, "p1")
    battle._available_switches, battle._force_switch = [reserve], True
    monkeypatch.setattr(
        player, "build_payoff_table", lambda *a: pytest.fail("no table on forced switch")
    )
    assert agent.choose_move(battle).order is reserve
    assert "forced replacement" in agent.last_decision.explanation


def test_failed_damage_estimates_still_choose_a_legal_move(agent, monkeypatch):
    battle = make_battle(moves=("splash",))
    monkeypatch.setattr(player, "flat_beliefs", lambda b: {})
    monkeypatch.setattr(player, "estimate_move_damage", lambda b: {})
    assert agent.choose_move(battle).order.id == "splash"
    assert "first legal move" in agent.last_decision.explanation


def test_empty_options_request_server_default(agent):
    battle = make_battle(moves=())
    assert agent.choose_move(battle).message == "/choose default"
    assert agent.decision_counts["default"] == 1
    assert agent.last_decision is None


def test_cosmetic_form_can_use_base_species_prior():
    battle = make_battle()
    battle._opponent_team = {"p2: Cosmetic": Pokemon(gen=9, species="pikachuhoenn")}
    battle.opponent_team["p2: Cosmetic"]._active = True
    assert flat_beliefs(battle)["pikachuhoenn"].move_probs


def test_unexpected_programming_error_is_not_hidden(agent, monkeypatch):
    def bug(*args):
        raise TypeError("unexpected bug")

    monkeypatch.setattr(player, "flat_beliefs", bug)
    with pytest.raises(TypeError, match="unexpected bug"):
        agent.choose_move(make_battle())


def test_invalid_strategy_rejected_before_connecting():
    with pytest.raises(ValueError, match="strategy"):
        ExplainableAgent(strategy="typo", start_listening=False)


def test_a_tera_decision_becomes_a_tera_order(agent):
    battle = make_battle("Blissey", "Blissey", ("seismictoss",))
    battle._can_tera = True
    order = agent._order(battle, Action("move", "seismictoss", tera=True))
    assert order.terastallize and order.message.endswith("terastallize")
    battle._can_tera = False  # used already: a tera decision is no longer legal
    with pytest.raises(ValueError):
        agent._order(battle, Action("move", "seismictoss", tera=True))
