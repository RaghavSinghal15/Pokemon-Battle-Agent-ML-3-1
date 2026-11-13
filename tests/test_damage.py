"""Damage-helper checks use real poke-env objects, without a running server."""

import logging
from copy import deepcopy

import pytest
from poke_env.battle import Battle, Move

from pokeagent.decision import damage
from pokeagent.decision.damage import estimate_move_damage, random_battle_stats


def make_battle(attacker="Gengar", defender="Blissey", moves=("seismictoss",)):
    """Build the same public switch events and own stats a live battle supplies."""
    battle = Battle("battle-test", "Alice", logging.getLogger(__name__), gen=9)
    for line in [
        "|player|p1|Alice|",
        "|player|p2|Bob|",
        f"|switch|p1a: Attacker|{attacker}, L100|100/100",
        f"|switch|p2a: Defender|{defender}, L100|100/100",
    ]:
        battle.parse_message(line.split("|"))
    battle.active_pokemon.stats = random_battle_stats(battle.active_pokemon.species, 100)
    battle._available_moves = [Move(move, gen=9) for move in moves]
    return battle


def test_default_stats_match_hand_calculated_level_100_pikachu():
    # HP: 2*35 + 31 + floor(85/4) + 100 + 10. Other stats use +5 instead of +110.
    assert random_battle_stats("pikachu", 100) == {
        "hp": 232,
        "atk": 167,
        "def": 137,
        "spa": 157,
        "spd": 157,
        "spe": 237,
    }


@pytest.mark.parametrize("level", [0, 101])
def test_invalid_level(level):
    with pytest.raises(ValueError):
        random_battle_stats("pikachu", level)


def test_fixed_damage_uses_estimated_maximum_hp_without_mutation():
    battle = make_battle()
    target = battle.opponent_active_pokemon
    before = deepcopy(target.stats)
    result = estimate_move_damage(battle)["seismictoss"]
    # Level-100 Seismic Toss deals exactly 100 HP; default Blissey HP = 672.
    assert result.error is None
    assert result.min_percent == pytest.approx(10000 / 672)
    assert result.max_percent == result.expected_percent == result.min_percent
    assert target.stats == before
    assert target is battle.opponent_active_pokemon
    assert target.current_hp_fraction == 1


def test_damaging_move_status_and_type_immunity():
    battle = make_battle("Pikachu", "Garchomp", ("thunderbolt", "surf", "protect"))
    estimates = estimate_move_damage(battle)
    assert all(result.error is None for result in estimates.values())
    assert estimates["thunderbolt"].expected_percent == 0  # Ground is immune to Electric.
    assert estimates["protect"].expected_percent == 0
    surf = estimates["surf"]
    assert 0 < surf.min_percent <= surf.expected_percent <= surf.max_percent


def test_revealed_ability_is_used():
    battle = make_battle("Pikachu", "Vaporeon", ("surf",))
    assert estimate_move_damage(battle)["surf"].expected_percent > 0
    battle.opponent_active_pokemon.ability = "waterabsorb"
    assert estimate_move_damage(battle)["surf"].expected_percent == 0


def test_revealed_item_and_boosts_are_used_and_preserved():
    battle = make_battle("Pikachu", "Blissey", ("thunderbolt",))
    normal = estimate_move_damage(battle)["thunderbolt"].expected_percent
    target = battle.opponent_active_pokemon
    target.item = "assaultvest"
    assert estimate_move_damage(battle)["thunderbolt"].expected_percent < normal
    target.boosts["spd"] = 2
    boosted = estimate_move_damage(battle)["thunderbolt"].expected_percent
    assert boosted < normal
    assert target.item == "assaultvest" and target.boosts["spd"] == 2


def test_known_stats_are_not_overwritten():
    battle = make_battle()
    battle.opponent_active_pokemon.stats["hp"] = 800
    estimate = estimate_move_damage(battle)["seismictoss"]
    assert estimate.expected_percent == pytest.approx(12.5)
    assert battle.opponent_active_pokemon.stats["hp"] == 800


def test_estimate_is_not_capped_at_remaining_hp():
    battle = make_battle()
    battle.opponent_active_pokemon.set_hp("1/100")
    assert estimate_move_damage(battle)["seismictoss"].expected_percent > 1


@pytest.mark.parametrize("state", ["forced", "missing", "fainted", "no_moves"])
def test_no_normal_move_decision(state):
    battle = make_battle()
    if state == "forced":
        battle._force_switch = True
    elif state == "missing":
        battle._opponent_team.clear()
    elif state == "fainted":
        battle.opponent_active_pokemon.faint()
    else:
        battle._available_moves = []
    assert estimate_move_damage(battle) == {}


def test_failures_are_not_silently_reported_as_zero(monkeypatch):
    battle = make_battle(moves=("seismictoss", "protect"))
    real_calculator = damage.calculate_damage

    def failing(attacker, defender, move, calculation):
        if move.id == "seismictoss":
            raise NotImplementedError("unsupported case")
        return real_calculator(attacker, defender, move, calculation)

    monkeypatch.setattr(damage, "calculate_damage", failing)
    estimates = estimate_move_damage(battle)
    assert estimates["seismictoss"].expected_percent is None
    assert "unsupported case" in estimates["seismictoss"].error
    assert estimates["protect"].expected_percent == 0


@pytest.mark.parametrize("result", [(-1, 2), (3, 2), (float("nan"), 2)])
def test_invalid_calculator_ranges_are_errors(monkeypatch, result):
    monkeypatch.setattr(damage, "calculate_damage", lambda *args: result)
    assert estimate_move_damage(make_battle())["seismictoss"].expected_percent is None


@pytest.mark.parametrize("hp", [0, -1, float("nan")])
def test_invalid_maximum_hp_is_an_explicit_error(hp):
    battle = make_battle()
    battle.opponent_active_pokemon.stats["hp"] = hp
    assert estimate_move_damage(battle)["seismictoss"].error is not None


def test_preparation_failure_is_reported_for_every_move(monkeypatch):
    battle = make_battle(moves=("seismictoss", "protect"))

    def unavailable(*args):
        raise KeyError("unknown species")

    monkeypatch.setattr(damage, "random_battle_stats", unavailable)
    results = estimate_move_damage(battle)
    assert set(results) == {"seismictoss", "protect"}
    assert all(result.expected_percent is None and result.error for result in results.values())


def test_other_generations_are_rejected():
    battle = Battle("battle-test", "Alice", logging.getLogger(__name__), gen=8)
    with pytest.raises(ValueError, match="Gen 9 singles"):
        estimate_move_damage(battle)


def test_recharge_is_a_rest_turn_not_a_calculator_failure(monkeypatch):
    def must_not_calculate(*args):
        raise AssertionError("Recharge must not enter the damage calculator")

    monkeypatch.setattr(damage, "calculate_damage", must_not_calculate)
    result = estimate_move_damage(make_battle(moves=("recharge",)))["recharge"]
    assert result.error is None
    assert result.min_percent == result.max_percent == result.expected_percent == 0
