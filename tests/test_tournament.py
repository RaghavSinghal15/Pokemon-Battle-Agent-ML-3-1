"""Controlled fake battle outcomes check bookkeeping, persistence and failure paths."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from pokeagent.eval import tournament


def result(won, finished=True):
    return SimpleNamespace(won=won, lost=None if won is None else not won, finished=finished)


class FakePlayer:
    format = "gen9randombattle"

    def __init__(self, outcomes=()):
        self.battles = {}
        self.outcomes = outcomes
        self.calls = 0

    async def battle_against(self, opponent, n_battles):
        self.calls += 1
        for i, won in enumerate(self.outcomes):
            key = f"game-{self.calls}-{i}"
            self.battles[key] = result(won)
            opponent.battles[key] = result(None if won is None else not won)


@pytest.fixture(autouse=True)
def fixed_source(monkeypatch):
    monkeypatch.setattr(
        tournament, "source_revision", lambda: {"commit": "test-sha", "tracked_changes": False}
    )


def run(player, opponent, path, games=3, **kwargs):
    return asyncio.run(tournament.run_matchup(player, opponent, games, output_path=path, **kwargs))


def test_fresh_results_ties_and_saved_provenance(tmp_path):
    player, opponent = FakePlayer([True, False, None]), FakePlayer()
    player.battles["old"] = result(True)
    opponent.battles["old"] = result(False)
    path = tmp_path / "result.json"
    report = run(player, opponent, path)
    assert (report["wins"], report["losses"], report["ties"], report["games"]) == (1, 1, 1, 3)
    assert report["win_rate"] == pytest.approx(1 / 3)
    saved = json.loads(path.read_text())
    assert saved["source"]["commit"] == "test-sha"
    assert saved["seed"] == 464
    assert saved["poke_env_version"]
    assert saved["started_utc"].endswith("+00:00")
    second = run(player, opponent, tmp_path / "second.json")
    assert second["games"] == 3 and second["wins"] == 1


def test_existing_result_is_not_overwritten(tmp_path):
    path = tmp_path / "exists.json"
    path.write_text("keep")
    player = FakePlayer([True])
    with pytest.raises(FileExistsError):
        run(player, FakePlayer(), path, 1)
    assert player.calls == 0 and path.read_text() == "keep"


@pytest.mark.parametrize("games", [0, -1, True, 1.5])
def test_bad_game_count(tmp_path, games):
    with pytest.raises(ValueError):
        run(FakePlayer(), FakePlayer(), tmp_path / "invalid.json", games)


def test_partial_completion_does_not_publish_success(tmp_path):
    path = tmp_path / "partial.json"
    with pytest.raises(RuntimeError, match="counts disagree"):
        run(FakePlayer([True]), FakePlayer(), path, 3)
    assert not path.exists()


def test_same_player_and_unfinished_games_are_rejected(tmp_path):
    player = FakePlayer()
    with pytest.raises(ValueError):
        run(player, player, tmp_path / "same.json")
    player.battles["running"] = result(None, finished=False)
    with pytest.raises(ValueError, match="unfinished"):
        run(player, FakePlayer(), tmp_path / "busy.json")


def test_battle_error_propagates_without_result(tmp_path):
    class FailedPlayer(FakePlayer):
        async def battle_against(self, opponent, n_battles):
            raise ConnectionError("server offline")

    path = tmp_path / "error.json"
    with pytest.raises(ConnectionError):
        run(FailedPlayer(), FakePlayer(), path)
    assert not path.exists()


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed(tmp_path, seed):
    with pytest.raises(ValueError):
        run(FakePlayer(), FakePlayer(), tmp_path / "invalid.json", seed=seed)


def test_timeout_has_no_success_file(tmp_path):
    class SlowPlayer(FakePlayer):
        async def battle_against(self, opponent, n_battles):
            await asyncio.sleep(1)

    path = tmp_path / "timeout.json"
    with pytest.raises(TimeoutError):
        run(SlowPlayer(), FakePlayer(), path, timeout=0.01)
    assert not path.exists()


def test_opponent_disagreement_is_rejected(tmp_path):
    class InconsistentPlayer(FakePlayer):
        async def battle_against(self, opponent, n_battles):
            await super().battle_against(opponent, n_battles)
            for key in opponent.battles:
                opponent.battles[key] = result(True)

    with pytest.raises(RuntimeError, match="records disagree"):
        run(InconsistentPlayer([True]), FakePlayer(), tmp_path / "bad.json", 1)
