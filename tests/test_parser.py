"""Exercise public turn boundaries, labels, live parity, and the replay export."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pokeagent.interfaces import Action, PublicState
from pokeagent.parser.export import export_decision_points
from pokeagent.parser.live import iter_live_states, state_from_battle
from pokeagent.parser.replay import iter_public_states, parse_battle

START = """|player|p1|Alice|1|1500
|player|p2|Bob|2|1600
|teamsize|p1|6
|teamsize|p2|6
|switch|p1a: Sparky|Pikachu, L90|100/100
|switch|p2a: Corviknight|Corviknight, L80|100/100
|turn|1"""


def test_turn_snapshot_is_independent_of_future_log_lines():
    suffix = """|-terastallize|p1a: Sparky|Electric
|move|p1a: Sparky|Thunderbolt|p2a: Corviknight
|-damage|p2a: Corviknight|51/100 par
|-item|p2a: Corviknight|Leftovers
|-ability|p2a: Corviknight|Pressure
|move|p2a: Corviknight|Roost|p2a: Corviknight
|upkeep
|turn|2
|-enditem|p2a: Corviknight|Leftovers
|turn|3
|win|Alice"""
    log = f"{START}\n{suffix}"
    states = list(iter_public_states("a", log.splitlines()))
    for turn in (1, 2, 3):
        prefix = log.split(f"|turn|{turn}", 1)[0] + f"|turn|{turn}"
        assert list(iter_public_states("a", prefix.splitlines())) == states[: 2 * turn]
    first, second, third = states[::2]
    assert first.me.rating == 1500
    assert first.opp.rating == 1600
    assert first.me.active.species == "pikachu"
    assert first.me.active.revealed_moves == []
    assert first.opp.active.hp_fraction == 1
    assert first.opp.active.revealed_item is None
    assert not first.opp.active.item_removed
    assert second.me.tera_used
    assert second.me.active.revealed_moves == ["thunderbolt"]
    assert second.opp.active.hp_fraction == 0.51
    assert second.opp.active.status == "par"
    assert second.opp.active.revealed_item == "leftovers"
    assert second.opp.active.revealed_ability == "pressure"
    assert third.opp.active.item_removed
    assert third.opp.active.revealed_item is None
    assert second.my_last_action == Action("move", "thunderbolt", tera=True)
    assert states[3].opp_last_action == second.my_last_action
    points = parse_battle("a", log, None, {})
    assert [(p.action.name, p.won) for p in points] == [("thunderbolt", True), ("roost", False)]


@pytest.mark.parametrize(
    "hidden",
    [
        "|cant|p2a: Corviknight|par",
        "|-damage|p2a: Corviknight|72/100|[from] confusion",
        "|move|p2a: Corviknight|Outrage|p1a: Sparky|[from]lockedmove",
        "|faint|p2a: Corviknight",
    ],
)
def test_hidden_choices_and_forced_replacements_do_not_label(hidden):
    log = f"""{START}
|move|p1a: Sparky|Thunderbolt|p2a: Corviknight
{hidden}
|upkeep
|switch|p2a: Charizard|Charizard, L85|100/100
|turn|2
|tie"""
    points = parse_battle("a", log, None, {})
    assert [(p.state.turn, p.state.perspective, p.action) for p in points] == [
        (1, "p1", Action("move", "thunderbolt"))
    ]


def test_voluntary_switch_and_called_move_labels():
    log = f"""{START}
|switch|p1a: Raichu|Raichu, L86|100/100
|move|p2a: Corviknight|Sleep Talk|p2a: Corviknight
|move|p2a: Corviknight|Brave Bird|p1a: Raichu|[from] move: Sleep Talk
|win|Bob"""
    points = parse_battle("a", log, None, {})
    assert [p.action for p in points] == [Action("switch", "raichu"), Action("move", "sleeptalk")]


def test_live_rounds_hp_and_skips_private_requests():
    spectator = f"{START}\n|-damage|p1a: Sparky|99/100\n|turn|2"
    live = spectator.replace("99/100", "299/300")
    live += '\n|request|{"private":true}\n|move|p1a: Sparky|Surf|p2a: Corviknight'
    replay_state = list(iter_public_states("a", spectator.splitlines()))[-2]
    battle = SimpleNamespace(
        player_role="p1",
        battle_tag="a",
        turn=2,
        _replay_data=[line.split("|") for line in live.splitlines()],
    )
    assert state_from_battle(battle) == replay_state
    split = spectator.replace(
        "|-damage|p1a: Sparky|99/100",
        "|split|p1\n|-damage|p1a: Sparky|299/300\n|-damage|p1a: Sparky|99/100",
    )
    assert list(iter_public_states("a", split.splitlines()))[-2] == replay_state


def test_form_change_keeps_species_at_first_reveal():
    start = START.replace("Pikachu, L90", "Palafin, L90")
    log = f"{start}\n|detailschange|p1a: Sparky|Palafin-Hero, L90\n|turn|2"
    assert list(iter_public_states("a", log.splitlines()))[-2].me.active.species == "palafin"


def test_export_writes_schema_and_counts_unsupported_battles(tmp_path):
    source = tmp_path / "battles.parquet"
    output = tmp_path / "decision_points.parquet"
    choice = "|move|p1a: Sparky|Thunderbolt|p2a: Corviknight"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"battle_id": "good", "log": f"{START}\n{choice}\n|win|Alice"},
                {"battle_id": "bad", "log": f"{START}\n{choice}\n|unexpected-event|"},
            ]
        ),
        source,
    )
    summary = export_decision_points(source, output)
    assert summary.battles == 2
    assert summary.rows == 1
    assert summary.errors == {"unexpected-event": 1}
    (row,) = pq.read_table(output).to_pylist()
    assert row["action"] == {"kind": "move", "name": "thunderbolt", "tera": False}
    assert row["won"] is True
    assert PublicState.from_dict(json.loads(row["state"])).battle_id == "good"
    summary = export_decision_points(source, output, battle_ids={"good"})
    assert (summary.battles, summary.rows, summary.errors) == (1, 1, {})


@pytest.mark.showdown
def test_parity_on_recorded_local_battles():
    capture = os.environ.get("POKEAGENT_PARSER_CAPTURE")
    if not capture:
        pytest.skip("Set POKEAGENT_PARSER_CAPTURE to the 1,000-game local capture")
    checked = 0
    games = 0
    with Path(capture).open(encoding="utf-8") as source:
        for line in source:
            battle = json.loads(line)
            games += 1
            states = list(iter_public_states(battle["battle_id"], battle["spectator_log"]))
            for side in ("p1", "p2"):
                expected = [state for state in states if state.perspective == side]
                actual = list(iter_live_states(battle["battle_id"], battle["streams"][side], side))
                assert expected == actual, (battle["battle_id"], side)
                checked += len(actual)
    assert games >= 1_000
    assert checked > 0
