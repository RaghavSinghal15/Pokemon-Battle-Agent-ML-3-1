"""Play styles (E7, #80): profiles, leave-one-out style features and clustering."""

import numpy as np
import pandas as pd
import pytest

from pokeagent.models.styles import (
    STYLE_STATS,
    action_kind,
    cluster,
    game_profiles,
    player_ids,
    player_profiles,
    style_features,
)


def test_action_kinds():
    assert action_kind({"kind": "switch", "name": "garchomp"}) == "switch"
    assert action_kind({"kind": "move", "name": "earthquake"}) == "attack"
    assert action_kind({"kind": "move", "name": "swordsdance"}) == "setup"
    assert action_kind({"kind": "move", "name": "stealthrock"}) == "hazard"
    assert action_kind({"kind": "move", "name": "protect"}) == "protect"
    assert action_kind({"kind": "move", "name": "thunderwave"}) == "other_status"


def test_player_ids_from_logs():
    log = "|j|☆Ash K.\n|player|p1|Ash K.|1|1500\n|player|p2|Misty|2|\n"
    battles = pd.DataFrame({"battle_id": ["b1"], "log": [log]})
    assert player_ids(battles) == {("b1", "p1"): "ashk", ("b1", "p2"): "misty"}


def profiles():
    def act(kind, name):
        return {"kind": kind, "name": name}

    points = pd.DataFrame(
        {
            "battle_id": ["b1"] * 4 + ["b2"] * 2 + ["b3"] * 2,
            "perspective": ["p1"] * 8,
            "action": [
                act("switch", "x"),
                act("switch", "y"),
                act("move", "earthquake"),
                act("move", "swordsdance"),
                act("move", "earthquake"),
                act("move", "earthquake"),
                act("switch", "x"),
                act("move", "earthquake"),
            ],
        }
    )
    table = game_profiles(points)
    table["player"] = "ash"
    return table


def test_game_and_player_profiles_are_decision_weighted():
    table = profiles()
    b1 = table.set_index("battle_id").loc["b1"]
    assert b1["n"] == 4 and b1["switch"] == pytest.approx(0.5) and b1["setup"] == 0.25
    player = player_profiles(table).loc["ash"]
    assert player["games"] == 3 and player["switch"] == pytest.approx(3 / 8)


def test_style_features_leave_out_the_current_game():
    train = profiles()
    rows = pd.DataFrame({"battle_id": ["b1", "b9", "b9"], "player": ["ash", "ash", "nobody"]})
    styles = style_features(rows, train)
    # b1's own two switches are excluded: other games b2, b3 have 1 switch in 4 decisions.
    assert styles.loc[0, "style_switch"] == pytest.approx(1 / 4)
    assert styles.loc[0, "style_known"] == 1
    # A validation game (b9) uses all of the player's training games.
    assert styles.loc[1, "style_switch"] == pytest.approx(3 / 8)
    # Unknown players get the training average and a 0 flag.
    assert styles.loc[2, "style_known"] == 0
    assert styles.loc[2, "style_switch"] == pytest.approx(3 / 8)


def test_cluster_finds_two_obvious_styles():
    rng = np.random.default_rng(0)
    switchers = rng.normal([0.4, 0.4, 0.05, 0.05, 0.05, 0.05], 0.02, size=(40, 6))
    attackers = rng.normal([0.05, 0.85, 0.02, 0.02, 0.03, 0.03], 0.02, size=(40, 6))
    players = pd.DataFrame(np.vstack([switchers, attackers]), columns=list(STYLE_STATS))
    result = cluster(players, ks=range(2, 5))
    assert result["chosen_k"] == 2 and sorted(result["sizes"]) == [40, 40]
    assert result["pca_coordinates"].shape == (80, 2)


def test_same_player_on_both_sides_leaves_out_the_whole_battle():
    train = pd.concat([profiles(), profiles().iloc[:1].assign(perspective="p2")])
    rows = pd.DataFrame({"battle_id": ["b1"], "player": ["ash"]})
    assert style_features(rows, train).loc[0, "style_switch"] == pytest.approx(1 / 4)
