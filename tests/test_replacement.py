"""Forced replacements are chosen by next-turn matchup, not by remaining HP (F7, #61).

The fake battle: our active Pokemon has just fainted to a Fire-type (Arcanine, known to
carry Flamethrower). We can send in a half-HP Water-type that resists Fire and hits it
super effectively, or a full-HP Grass-type that Fire hits super effectively.
"""

import json

import pytest
from poke_env.battle import Pokemon, SideCondition
from test_damage import make_battle

from pokeagent.agent import player
from pokeagent.agent.player import ExplainableAgent
from pokeagent.config import BATTLE_FORMAT
from pokeagent.decision.damage import random_battle_stats
from pokeagent.decision.payoff import PayoffUnavailable, replacement_tables
from pokeagent.interfaces import SetBelief

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(autouse=True)
def public_sets(tmp_path, monkeypatch):
    """A tiny stand-in sets file, so these tests need no Showdown checkout (as in CI).

    The agent only checks that the file exists; the tests pass their own beliefs.
    """
    path = tmp_path / "replacement-sets.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(player, "RANDOM_BATTLE_SETS_PATH", path)


ARCANINE = {"arcanine": SetBelief("arcanine", move_probs={"flamethrower": 1.0})}


def reserve(battle, species, hp, moves):
    """Add one of our own reserves, with the stats and moves our team request would show."""
    mon = Pokemon(gen=9, species=species)
    mon.set_hp(f"{hp}/100")
    mon.stats = random_battle_stats(species, 100)
    for move in moves:
        mon._add_move(move)
    battle.team[f"p1: {species}"] = mon
    return mon


def faint_battle():
    battle = make_battle("Gengar", "Arcanine", moves=())
    battle.active_pokemon.faint()
    battle._team_size = {"p1": 3, "p2": 1}
    water = reserve(battle, "vaporeon", 50, ["surf"])
    grass = reserve(battle, "venusaur", 100, ["gigadrain"])
    battle._available_switches, battle._force_switch = [grass, water], True
    return battle


def test_tables_score_each_reserve_with_its_own_moves():
    tables = replacement_tables(faint_battle(), ARCANINE, None)
    assert set(tables) == {"vaporeon", "venusaur"}
    assert [a.name for a in tables["vaporeon"].our_actions] == ["surf"]
    assert [a.name for a in tables["venusaur"].our_actions] == ["gigadrain"]
    # Surf (2x) against Flamethrower (0.5x) beats Giga Drain (0.5x) against Flamethrower (2x).
    assert tables["vaporeon"].values[0][0] > tables["venusaur"].values[0][0]


@pytest.mark.parametrize("strategy", ["best_response", "lp"])
def test_half_hp_resist_beats_full_hp_weakness(strategy, monkeypatch):
    monkeypatch.setattr(player, "flat_beliefs", lambda battle: ARCANINE)
    agent = ExplainableAgent(battle_format=BATTLE_FORMAT, strategy=strategy, start_listening=False)
    order = agent.choose_move(faint_battle())
    assert order.order.species == "vaporeon"
    assert agent.last_decision.strategy == strategy
    assert "by matchup" in agent.last_decision.explanation
    assert not agent.fallback_reasons


def test_stealth_rock_is_charged_on_entry():
    battle = faint_battle()
    before = replacement_tables(battle, ARCANINE, None)["vaporeon"].values[0][0]
    battle.side_conditions[SideCondition.STEALTH_ROCK] = 1
    after = replacement_tables(battle, ARCANINE, None)["vaporeon"].values[0][0]
    # Water is neutral to Rock: Stealth Rock takes 1/8 of maximum HP on the way in.
    assert after == pytest.approx(before - 1 / 8, abs=0.02)


def test_unscorable_replacement_falls_back_to_healthiest(monkeypatch):
    battle = faint_battle()
    battle.opponent_active_pokemon.faint()  # both fainted: their next Pokemon is unknown
    with pytest.raises(PayoffUnavailable):
        replacement_tables(battle, ARCANINE, None)
    monkeypatch.setattr(player, "flat_beliefs", lambda battle: ARCANINE)
    agent = ExplainableAgent(battle_format=BATTLE_FORMAT, start_listening=False)
    assert agent.choose_move(battle).order.species == "venusaur"
    assert "could not be scored" in agent.last_decision.explanation
