"""Hand-calculated examples plus real poke-env adapters for the F4 scoring proposal."""

import json
from copy import deepcopy

import pytest
from poke_env.battle import Field, Move, Pokemon, SideCondition, Status
from test_damage import make_battle

from pokeagent.decision import _payoff_state, payoff
from pokeagent.decision._payoff_state import entry_damage, first_probability, scenario_copy
from pokeagent.decision._turn import TurnEffect, turn_value
from pokeagent.decision.payoff import (
    SWITCH_COLUMN,
    PayoffApproximationWarning,
    PayoffUnavailable,
    build_payoff_table,
)
from pokeagent.interfaces import SetBelief


def attack(amount, chance=1):
    return TurnEffect((amount, amount), chance)


@pytest.mark.parametrize(
    "ours,theirs,hp,order,entry,expected",
    [
        (attack(0.4), attack(0), (1, 0.8), 1, (0, 0), 0.4),
        (attack(0.4), attack(0.25), (1, 1), 1, (0, 0), 0.15),
        (attack(0.5), attack(0.6), (0.2, 0.3), 1, (0, 0), 0.55),
        (attack(0.5), attack(0.6), (0.2, 0.3), 0, (0, 0), -0.45),
        (attack(0.5), attack(0.6), (0.2, 0.3), 0.5, (0, 0), 0.05),
        (attack(0.5, 0.8), attack(0.1), (1, 0.3), 1, (0, 0), 0.42),
        (TurnEffect((0.2, 0.4)), attack(0), (1, 0.3), 1, (0, 0), 0.375),
        (TurnEffect(heal=0.5), attack(0.2), (0.7, 1), 1, (0, 0), 0.1),
        (attack(0), attack(0.25), (1, 1), 0, (0.125, 0), -0.375),
        (attack(0.5), attack(0.6), (0.2, 0.3), 1, (0.3, 0), -0.45),
    ],
)
def test_worked_examples(ours, theirs, hp, order, entry, expected):
    assert turn_value(*hp, ours, theirs, order, entry) == pytest.approx(expected)


def test_ko_weight_sensitivity():
    values = [turn_value(1, 0.3, attack(0.5), attack(0), 1, ko_bonus=k) for k in (0, 0.25, 0.5)]
    assert values == pytest.approx([0.3, 0.55, 0.8])


def test_heal_cancelled_by_earlier_ko():
    assert turn_value(0.2, 1, TurnEffect(heal=0.5), attack(0.3), 0) == pytest.approx(-0.45)


def test_protect_and_bypass():
    assert turn_value(1, 1, TurnEffect(protect=True), attack(0.4), 1) == 0
    bypass = TurnEffect((0.4, 0.4), blocked_by_protect=False)
    assert turn_value(1, 1, TurnEffect(protect=True), bypass, 1) == pytest.approx(-0.4)


@pytest.mark.parametrize(
    "effect",
    [
        TurnEffect((-1, 1)),
        TurnEffect((1, float("nan"))),
        TurnEffect(hit_chance=2),
        TurnEffect(heal=-1),
    ],
)
def test_invalid_effect(effect):
    with pytest.raises(ValueError):
        turn_value(1, 1, effect, attack(0), 1)


def table_battle(moves=("seismictoss",), candidates=None):
    battle = make_battle("Blissey", "Blissey", moves)
    battle._team_size = {"p1": 1, "p2": 1}
    return battle, {"blissey": SetBelief("blissey", move_probs=candidates or {"seismictoss": 1})}


def build(battle, beliefs, probs=None, **options):
    with pytest.warns(PayoffApproximationWarning):
        return build_payoff_table(battle, beliefs, probs, **options)


def test_class_mass_mapping(monkeypatch):
    """Belief weights only (F15 off): Seismic Toss and Night Shade do the same damage anyway."""
    monkeypatch.setattr(payoff, "PAYOFF_ATTACK_TEMPERATURE", 0.0)
    battle, beliefs = table_battle(
        candidates={"seismictoss": 0.75, "nightshade": 0.25, "protect": 1}
    )
    battle._team_size["p2"] = 6
    table = build(battle, beliefs, {"attack": 0.6, "non_damaging": 0.1, "switch": 0.3})
    assert [a.name for a in table.opp_actions] == [
        "nightshade",
        "protect",
        "seismictoss",
        SWITCH_COLUMN,
    ]
    assert table.opp_probs == pytest.approx([0.15, 0.1, 0.45, 0.3])
    assert sum(table.opp_probs) == pytest.approx(1)


def test_revelation_overrides_prior_without_mutating_it():
    battle, beliefs = table_battle(candidates={"seismictoss": 0})
    battle.opponent_active_pokemon.moves["seismictoss"] = Move("seismictoss", gen=9)
    assert len(build(battle, beliefs).opp_actions) == 1
    assert beliefs["blissey"].move_probs["seismictoss"] == 0


def test_full_status_only_pool_removes_impossible_attack_mass():
    battle, beliefs = table_battle(candidates={"protect": 1})
    table = build(battle, beliefs, {"attack": 0.6, "non_damaging": 0.4, "switch": 0})
    assert table.opp_probs == [1.0]


@pytest.mark.parametrize(
    "probs",
    [
        {"attack": 1},
        {"attack": -0.1, "non_damaging": 1.1, "switch": 0},
        {"attack": 0.5, "non_damaging": 0, "switch": 0},
    ],
)
def test_bad_forecasts(probs):
    battle, beliefs = table_battle()
    with pytest.raises(PayoffUnavailable):
        build_payoff_table(battle, beliefs, probs)


def test_impossible_switch_forecast_is_renormalised():
    battle, beliefs = table_battle()
    table = build(battle, beliefs, {"attack": 0.8, "non_damaging": 0, "switch": 0.2})
    assert table.opp_probs == [1.0]


def test_real_damage_order_and_hp_capping():
    battle, beliefs = table_battle()
    for mon in (battle.active_pokemon, battle.opponent_active_pokemon):
        mon.stats = dict(battle.active_pokemon.stats, hp=400, spe=100)
        mon.set_hp("20/100")
    battle.active_pokemon.stats["spe"] = 200
    assert build(battle, beliefs).values[0][0] == pytest.approx(0.45)
    battle.active_pokemon.stats["spe"] = 50
    assert build(battle, beliefs).values[0][0] == pytest.approx(-0.45)


def add_reserve(battle, side):
    reserve = Pokemon(gen=9, species="snorlax")
    reserve.set_hp("100/100")
    reserve.stats = dict(battle.active_pokemon.stats, hp=400)
    team = battle.team if side == "p1" else battle.opponent_team
    team[f"{side}: Reserve"] = reserve
    return reserve


def test_unknown_and_known_switch_targets():
    """Without stand-ins (the default): an attack into an unseen switch-in scores nothing."""
    battle, beliefs = table_battle()
    battle._team_size["p2"] = 6
    assert build(battle, beliefs).values[0][-1] == 0
    add_reserve(battle, "p2")
    assert build(battle, beliefs).values[0][-1] == pytest.approx(0.25 / 5)


def sets_file(tmp_path, monkeypatch, species):
    """A tiny public sets file listing only these species (CI has no Showdown checkout)."""
    path = tmp_path / f"sets-{'-'.join(species)}.json"
    path.write_text(json.dumps({name: {"level": 80, "sets": []} for name in species}))
    monkeypatch.setattr(_payoff_state, "RANDOM_BATTLE_SETS_PATH", path)


def test_unseen_switch_in_is_scored_against_stand_ins(tmp_path, monkeypatch):
    """F13: Seismic Toss into an unseen switch-in now does damage to a stand-in."""
    battle, beliefs = table_battle()
    battle._team_size["p2"] = 6
    sets_file(tmp_path, monkeypatch, ["snorlax"])
    assert build(battle, beliefs, unseen_samples=8).values[0][-1] > 0
    assert build(battle, beliefs).values[0][-1] == 0  # off by default
    # Ghosts are immune to Seismic Toss: a pool of only Gengar gives no credit again.
    sets_file(tmp_path, monkeypatch, ["gengar"])
    assert build(battle, beliefs, unseen_samples=8).values[0][-1] == 0


def test_stand_ins_follow_the_species_clause_and_repeat_each_turn(tmp_path, monkeypatch):
    battle, _ = table_battle()
    sets_file(tmp_path, monkeypatch, ["blissey", "chansey", "snorlax", "gengar", "pikachu"])
    first = _payoff_state.unseen_stand_ins(battle, 3, seed=1)
    assert first.keys() == _payoff_state.unseen_stand_ins(battle, 3, seed=1).keys()
    species = {mon.species for mon in first.values()}
    assert len(species) == 3 and "blissey" not in species  # Blissey is already on their team
    for key, mon in first.items():
        assert battle.opponent_role is None or key.startswith(battle.opponent_role)
        assert mon.level == 80 and mon.current_hp_fraction == 1
    # Asking for more than the pool holds returns the whole pool.
    assert len(_payoff_state.unseen_stand_ins(battle, 10, seed=1)) == 4


def test_switch_incoming_target_and_hazards_once():
    battle, beliefs = table_battle(moves=("protect",))
    battle._available_switches = [add_reserve(battle, "p1")]
    battle.side_conditions[SideCondition.SPIKES] = 1
    table = build(battle, beliefs)
    assert table.our_actions[1].kind == "switch"
    assert table.values[1][0] == pytest.approx(-0.375)


def test_priority_trick_room_and_modifiers():
    battle, _ = table_battle()
    ours, theirs = Move("seismictoss", gen=9), Move("quickattack", gen=9)
    scenario = scenario_copy(battle, "p1: Attacker", "p2: Defender")
    scenario.active_pokemon.stats["spe"] = 200
    scenario.opponent_active_pokemon.stats["spe"] = 100
    assert first_probability(scenario, ours, theirs) == 0
    assert first_probability(scenario, ours, ours) == 1
    scenario.fields[Field.TRICK_ROOM] = 1
    assert first_probability(scenario, ours, ours) == 0
    scenario.fields.clear()
    scenario.opponent_active_pokemon.item = "choicescarf"
    scenario.opponent_side_conditions[SideCondition.TAILWIND] = 1
    assert first_probability(scenario, ours, ours) == 0


def test_hazard_protection():
    battle = make_battle("Pikachu", "Charizard")
    scenario = scenario_copy(battle, "p1: Attacker", "p2: Defender")
    mon = scenario.opponent_active_pokemon
    assert entry_damage(mon, scenario, {SideCondition.SPIKES: 3}) == 0
    assert entry_damage(mon, scenario, {SideCondition.STEALTH_ROCK: 1}) > 0.49
    mon.item = "heavydutyboots"
    assert entry_damage(mon, scenario, {SideCondition.STEALTH_ROCK: 1}) == 0


def test_failed_calculator_is_not_zero(monkeypatch):
    def fail(*args):
        raise KeyError("unsupported calculation")

    monkeypatch.setattr(_payoff_state, "calculate_damage", fail)
    battle, beliefs = table_battle()
    with pytest.raises(PayoffUnavailable, match="Damage calculation failed"):
        build_payoff_table(battle, beliefs, None)


def test_input_isolation():
    battle, beliefs = table_battle()
    before = deepcopy(
        (
            battle.active_pokemon.stats,
            battle.opponent_active_pokemon.stats,
            battle._replay_data,
            beliefs,
        )
    )
    forecast = {"attack": 1.0, "non_damaging": 0.0, "switch": 0.0}
    build(battle, beliefs, forecast)
    assert before == (
        battle.active_pokemon.stats,
        battle.opponent_active_pokemon.stats,
        battle._replay_data,
        beliefs,
    )
    assert forecast == {"attack": 1, "non_damaging": 0, "switch": 0}


def test_sleep_table_is_available():
    battle, beliefs = table_battle()
    battle.active_pokemon.status = Status.SLP
    assert build(battle, beliefs).values[0][0] < 0


def test_recharge_still_takes_reply_damage():
    battle, beliefs = table_battle(moves=("recharge",))
    assert build(battle, beliefs).values[0][0] < 0


def test_no_moves_cannot_silently_produce_a_switch_only_table():
    battle, _ = table_battle()
    battle._team_size["p2"] = 6
    with pytest.raises(PayoffUnavailable, match="No opponent move candidates"):
        build_payoff_table(battle, {}, None)


def test_real_protect_adapter_blocks_incoming_damage():
    battle, beliefs = table_battle(moves=("protect",))
    assert build(battle, beliefs).values[0][0] == 0


def test_current_tera_type_is_respected_without_adding_new_tera_rows():
    battle, beliefs = table_battle()
    baseline = build(battle, beliefs).values[0][0]
    battle.opponent_active_pokemon.terastallize("Ghost")
    table = build(battle, beliefs)
    assert table.values[0][0] < baseline
    assert not any(action.tera for action in table.our_actions)


def test_sleep_approximation_leaves_input_stats_untouched():
    battle, beliefs = table_battle()
    battle.active_pokemon.status = Status.SLP
    before = deepcopy(battle.opponent_active_pokemon.stats)
    build(battle, beliefs)
    assert before == battle.opponent_active_pokemon.stats


def test_scoring_is_symmetric_under_swapping_the_players():
    ours, theirs = attack(0.3, 0.7), attack(0.4, 0.8)
    value = turn_value(0.7, 0.4, ours, theirs, 0.3, (0.1, 0.05))
    reverse = turn_value(0.4, 0.7, theirs, ours, 0.7, (0.05, 0.1))
    assert value == pytest.approx(-reverse)


def test_belief_found_after_a_form_change():
    """The parser keeps the first form's name; poke-env follows form changes (#fallbacks)."""
    from pokeagent.decision.payoff import belief_for

    palafin = SetBelief("palafin", move_probs={"jetpunch": 1.0})
    assert belief_for("palafinhero", {"palafin": palafin}) is palafin
    assert belief_for("mimikyubusted", {"mimikyu": palafin}) is palafin
    assert belief_for("garchomp", {"palafin": palafin}) is None


def test_exact_move_weights_split_each_class():
    """E8: the class mass is shared by the exact-move model's weights, not belief inclusion."""
    battle, beliefs = table_battle(
        candidates={"seismictoss": 0.75, "nightshade": 0.25, "protect": 1}
    )
    forecast = {"attack": 0.6, "non_damaging": 0.4, "switch": 0.0}
    weights = {"seismictoss": 0.2, "nightshade": 0.6, "protect": 0.2}
    with pytest.warns(PayoffApproximationWarning):
        table = build_payoff_table(battle, beliefs, forecast, move_weights=weights)
    probs = dict(zip([a.name for a in table.opp_actions], table.opp_probs, strict=True))
    assert probs["nightshade"] == pytest.approx(0.6 * 0.6 / 0.8)
    assert probs["seismictoss"] == pytest.approx(0.6 * 0.2 / 0.8)
    assert probs["protect"] == pytest.approx(0.4)
    # A candidate the model did not score keeps a tiny weight rather than vanishing.
    with pytest.warns(PayoffApproximationWarning):
        table = build_payoff_table(battle, beliefs, forecast, move_weights={"nightshade": 1.0})
    probs = dict(zip([a.name for a in table.opp_actions], table.opp_probs, strict=True))
    assert 0 < probs["seismictoss"] < 0.001


def tera_battle(tera_type="Ghost"):
    """Blissey vs Blissey with Seismic Toss (Fighting), and tera still available."""
    battle, beliefs = table_battle()
    battle._can_tera = True
    battle.active_pokemon._last_request = {"teraType": tera_type}
    return battle, beliefs


def test_tera_type_is_offered_only_when_allowed():
    battle, _ = tera_battle()
    assert payoff.our_tera_type(battle) == "ghost"
    battle._can_tera = False
    assert payoff.our_tera_type(battle) is None
    battle, _ = tera_battle("Stellar")
    assert payoff.our_tera_type(battle) is None  # Stellar does not change the defensive type
    battle, _ = tera_battle()
    battle.active_pokemon.terastallize("ghost")
    assert payoff.our_tera_type(battle) is None  # already used


def test_tera_rows_score_the_new_type(monkeypatch):
    """F14: tera Ghost makes Blissey immune to Seismic Toss, so that row loses nothing."""
    monkeypatch.setattr(payoff, "PAYOFF_TERA_COST", 0.0)
    battle, beliefs = tera_battle()
    table = build(battle, beliefs)
    labels = [(a.name, a.tera) for a in table.our_actions]
    assert labels == [("seismictoss", False), ("seismictoss", True)]
    plain, tera = table.values
    assert tera[0] > plain[0]  # their Seismic Toss no longer hurts us
    # Without lasting effects, the gain is exactly the damage their Seismic Toss would do.
    monkeypatch.setattr(payoff, "PAYOFF_EFFECT_TURNS", 0.0)
    monkeypatch.setattr(payoff, "PAYOFF_HAZARD_WEIGHT", 0.0)
    plain, tera = build(battle, beliefs).values
    their_toss = 100 / battle.active_pokemon.stats["hp"]  # fixed damage = their level (100)
    assert tera[0] - plain[0] == pytest.approx(their_toss)


def test_tera_cost_is_charged_once_per_cell(monkeypatch):
    monkeypatch.setattr(payoff, "PAYOFF_EFFECT_TURNS", 0.0)
    monkeypatch.setattr(payoff, "PAYOFF_HAZARD_WEIGHT", 0.0)
    battle, beliefs = tera_battle()
    monkeypatch.setattr(payoff, "PAYOFF_TERA_COST", 0.0)
    free = build(battle, beliefs).values[1]
    monkeypatch.setattr(payoff, "PAYOFF_TERA_COST", 0.3)
    costly = build(battle, beliefs).values[1]
    assert costly == pytest.approx([v - 0.3 for v in free])


def test_attacks_that_hurt_more_are_expected_more(monkeypatch):
    """F15: a 4x super-effective attack gets more of the attack mass than a resisted one."""
    battle = make_battle("Charizard", "Garchomp", ("flamethrower",))
    battle._team_size = {"p1": 1, "p2": 1}
    beliefs = {"garchomp": SetBelief("garchomp", move_probs={"stoneedge": 0.5, "firefang": 0.5})}
    forecast = {"attack": 1.0, "non_damaging": 0.0, "switch": 0.0}

    def probabilities():
        table = build(battle, beliefs, forecast)
        names = [a.name for a in table.opp_actions]
        return dict(zip(names, table.opp_probs, strict=True))

    monkeypatch.setattr(payoff, "PAYOFF_ATTACK_TEMPERATURE", 0.0)
    flat = probabilities()
    assert flat["stoneedge"] == pytest.approx(flat["firefang"])
    monkeypatch.setattr(payoff, "PAYOFF_ATTACK_TEMPERATURE", 0.1)
    weighted = probabilities()
    # Stone Edge is 4x effective against Charizard; Fire Fang is resisted.
    assert weighted["stoneedge"] > 0.9
    assert sum(weighted.values()) == pytest.approx(1)
