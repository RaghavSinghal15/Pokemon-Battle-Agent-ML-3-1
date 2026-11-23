"""Ablation runner (A6, #35): variants, the lasting-effects switch and the report."""

import json
import pickle

import numpy as np
import pytest

from pokeagent.agent import player
from pokeagent.decision import payoff
from pokeagent.eval import ablations
from pokeagent.models.opponent import OpponentModel


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    sets = tmp_path / "sets.json"
    sets.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(player, "RANDOM_BATTLE_SETS_PATH", sets)
    model = OpponentModel("flat", class_frequencies=np.full(3, 1 / 3))
    for name in ("opponent.pkl", "naive.pkl", "value.pkl"):
        (tmp_path / name).write_bytes(pickle.dumps(model))
    monkeypatch.setattr(ablations, "OPPONENT_MODEL_PATH", tmp_path / "opponent.pkl")
    monkeypatch.setattr(ablations, "NAIVE_MODEL_PATH", tmp_path / "naive.pkl")
    monkeypatch.setattr(ablations, "VALUE_UTILITY_MODEL_PATH", tmp_path / "value.pkl")


@pytest.mark.parametrize("variant", list(ablations.VARIANTS))
def test_every_variant_builds_with_its_switch(variant, artifacts):
    agent = ablations.build_agent(variant, start_listening=False)
    assert agent.strategy == ("lp" if variant.startswith("lp") else "best_response")
    assert agent.utility == ("hp" if variant == "hp_utility" else "value")
    assert agent.belief_updates == (variant != "no_belief_updates")
    assert isinstance(agent, ablations.DamageOnlyAgent) == (variant == "damage_only")


def test_lasting_effects_switch_restores_the_weights():
    before = payoff.PAYOFF_EFFECT_TURNS, payoff.PAYOFF_HAZARD_WEIGHT
    with ablations.lasting_effects(False):
        assert payoff.PAYOFF_EFFECT_TURNS == payoff.PAYOFF_HAZARD_WEIGHT == 0
    assert (payoff.PAYOFF_EFFECT_TURNS, payoff.PAYOFF_HAZARD_WEIGHT) == before


def test_report_adds_up_chunks_per_variant(tmp_path):
    for i, (variant, wins) in enumerate([("full", 180), ("full", 190), ("lp", 160)]):
        record = {"variant": variant, "games": 250, "wins": wins, "source": {"commit": "abc1234ff"}}
        (tmp_path / f"{variant}-{i}.json").write_text(json.dumps(record), encoding="utf-8")
    rows = ablations.summarise(tmp_path)
    assert [(r["variant"], r["games"], r["wins"]) for r in rows] == [
        ("full", 500, 370),
        ("lp", 250, 160),
    ]
    text = ablations.markdown(rows)
    assert "| full |" in text and "-10.0 pts" in text and "abc1234" in text


def test_switched_off_restores_every_setting():
    from pokeagent.agent import player
    from pokeagent.decision import payoff

    before = (
        payoff.our_tera_type,
        payoff.PAYOFF_ATTACK_TEMPERATURE,
        player.PAYOFF_STAND_INS_FOR,
        payoff.PAYOFF_EFFECT_TURNS,
    )
    with ablations.switched_off("no_tera"):
        assert payoff.our_tera_type(object()) is None
    with ablations.switched_off("no_attack_weighting"):
        assert payoff.PAYOFF_ATTACK_TEMPERATURE == 0
    with ablations.switched_off("lp_no_stand_ins"):
        assert player.PAYOFF_STAND_INS_FOR == ()
    with ablations.switched_off("no_lasting_effects"):
        assert payoff.PAYOFF_EFFECT_TURNS == 0
    after = (
        payoff.our_tera_type,
        payoff.PAYOFF_ATTACK_TEMPERATURE,
        player.PAYOFF_STAND_INS_FOR,
        payoff.PAYOFF_EFFECT_TURNS,
    )
    assert before == after
