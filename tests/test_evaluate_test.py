"""Final test-split evaluation (E5, #38): the bootstrap and the run-once guard."""

import numpy as np
import pytest

from pokeagent.models import evaluate_test


def test_paired_bootstrap_resamples_whole_games():
    # Model A is better by 0.1 on every row of game "a" and equal on game "b".
    losses_a = np.array([0.5, 0.5, 0.7, 0.7])
    losses_b = np.array([0.6, 0.6, 0.7, 0.7])
    games = np.array(["a", "a", "b", "b"])
    result = evaluate_test.paired_bootstrap(losses_a, losses_b, games, rounds=200)
    assert result["difference"] == pytest.approx(-0.05)
    low, high = result["ci95"]
    # Only two games: every resample is all-a (-0.1), all-b (0) or a mix (-0.05).
    assert -0.1 - 1e-9 <= low <= high <= 1e-9


def test_refuses_to_run_twice(tmp_path, monkeypatch):
    report = tmp_path / "test_evaluation.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate_test, "REPORT_JSON", report)
    with pytest.raises(SystemExit, match="already exists"):
        evaluate_test.main()
