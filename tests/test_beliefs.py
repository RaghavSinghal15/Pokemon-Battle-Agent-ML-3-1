"""Hand-check priors and updates against small subsets of Showdown's set schema."""

import json

import pytest
from test_interfaces import make_state

from pokeagent.beliefs import tracker
from pokeagent.interfaces import PokemonView


def _template(role, moves, abilities, tera_types):
    return {
        "role": role,
        "movepool": moves.split(", "),
        "abilities": abilities.split(", "),
        "teraTypes": tera_types.split(", "),
    }


@pytest.fixture(autouse=True)
def set_data(tmp_path, monkeypatch):
    path = tmp_path / "sets.json"
    species = {
        "charizard": [
            _template(
                "Fast Attacker",
                "Earthquake, Hurricane, Focus Blast, Flamethrower, Roost",
                "Blaze, Solar Power",
                "Fire, Ground",
            ),
            _template(
                "Setup Sweeper", "Earthquake, Dragon Dance, Flare Blitz, Outrage", "Blaze", "Ground"
            ),
        ],
        "ditto": [_template("Fast Support", "Transform", "Imposter", "Normal")],
        "blastoise": [
            _template(
                "Setup Sweeper",
                "Shell Smash, Ice Beam, Hydro Pump, Earthquake",
                "Torrent",
                "Ground, Water",
            ),
            _template(
                "Setup Sweeper", "Shell Smash, Ice Beam, Hydro Pump, Surf", "Torrent", "Water"
            ),
        ],
        "garchomp": [_template("Fast Support", "Spikes", "Rough Skin", "Ground")],
        "corviknight": [_template("Bulky Support", "Roost", "Pressure", "Water")],
    }
    path.write_text(
        json.dumps({name: {"sets": sets} for name, sets in species.items()}), encoding="utf-8"
    )
    monkeypatch.setattr(tracker, "RANDOM_BATTLE_SETS_PATH", path)


def test_charizard_prior():
    belief = tracker.prior_belief("Charizard")
    assert belief.species == "charizard"
    assert belief.role_probs == {"Fast Attacker": 0.5, "Setup Sweeper": 0.5}
    assert belief.move_probs["earthquake"] == pytest.approx(0.9)
    assert belief.move_probs["hurricane"] == pytest.approx(0.4)
    assert sum(belief.move_probs.values()) == pytest.approx(4)
    assert belief.ability_probs == {"blaze": 0.75, "solarpower": 0.25}
    assert belief.tera_probs == {"fire": 0.25, "ground": 0.75}
    assert belief.item_probs == {}


def test_charizard_shared_move_reweights_roles():
    prior = tracker.prior_belief("charizard")
    seen = PokemonView("charizard", 85, revealed_moves=["Earthquake", "earthquake"])
    result = tracker.update_belief(prior, seen)
    # Earthquake appears in four of five attacker sets and the only sweeper set.
    assert result.role_probs == pytest.approx({"Fast Attacker": 4 / 9, "Setup Sweeper": 5 / 9})
    assert result.move_probs["earthquake"] == 1
    assert result.move_probs["hurricane"] == pytest.approx(1 / 3)
    assert sum(result.move_probs.values()) == pytest.approx(4)
    assert prior.move_probs["earthquake"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "evidence,role",
    [
        ({"revealed_moves": ["Dragon Dance"]}, "Setup Sweeper"),
        ({"revealed_ability": "Solar Power"}, "Fast Attacker"),
        ({"tera_type": "Fire", "terastallized": True}, "Fast Attacker"),
    ],
)
def test_charizard_exclusive_evidence_eliminates_roles(evidence, role):
    result = tracker.update_belief(
        tracker.prior_belief("charizard"), PokemonView("charizard", 85, **evidence)
    )
    assert result.role_probs == {role: 1}
    assert sum(result.ability_probs.values()) == pytest.approx(1)
    assert sum(result.tera_probs.values()) == pytest.approx(1)
    if "revealed_ability" in evidence:
        assert result.ability_probs == {"solarpower": 1}
    if "tera_type" in evidence:
        assert result.tera_probs == {"fire": 1}


def test_ditto_item_reveal_and_removal():
    prior = tracker.prior_belief("ditto")
    seen = PokemonView("ditto", 100, revealed_moves=["Transform"], revealed_item="Choice Scarf")
    result = tracker.update_belief(prior, seen)
    assert result.move_probs == {"transform": 1}
    assert result.item_probs == {"choicescarf": 1}
    seen.item_removed = True
    assert tracker.update_belief(prior, seen).item_probs == {}


def test_blastoise_templates_with_the_same_role_keep_separate_pools():
    prior = tracker.prior_belief("blastoise")
    assert prior.role_probs == {"Setup Sweeper": 1}
    assert prior.move_probs["surf"] == 0.5
    result = tracker.update_belief(prior, PokemonView("blastoise", 80, revealed_moves=["Surf"]))
    assert result.move_probs["surf"] == 1
    assert "earthquake" not in result.move_probs
    assert result.tera_probs == {"water": 1}


def test_unmodelled_move_preserves_observations_and_prior_roles():
    prior = tracker.prior_belief("ditto")
    seen = PokemonView("ditto", 100, revealed_moves=["Transform", "Surf"])
    result = tracker.update_belief(prior, seen)
    assert result.role_probs == prior.role_probs
    assert result.move_probs == {"transform": 1, "surf": 1}


def test_unknown_species_and_placeholder_item():
    prior = tracker.prior_belief("unmodelled")
    seen = PokemonView("unmodelled", 80, revealed_item="unknown_item")
    assert tracker.update_belief(prior, seen) == prior


def test_beliefs_for_uses_the_revealed_opponent_team_only():
    beliefs = tracker.beliefs_for(make_state())
    assert beliefs["garchomp"].move_probs == {"spikes": 1}
    assert beliefs["corviknight"].move_probs == {"roost": 1}
    assert set(beliefs) == {"garchomp", "corviknight"}
