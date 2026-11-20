"""Check that malformed spectator captures never become replay archives."""

from types import SimpleNamespace

import pytest

from pokeagent.eval.replays import write_public_logs


@pytest.mark.parametrize(
    "lines,error",
    [
        (["|init|battle", "|turn|8"], "Incomplete or inconsistent"),
        (["|init|battle", "|tie"], "Incomplete or inconsistent"),
        (["|init|battle", "|request|{}", "|win|Alice"], "Private protocol"),
        (["|init|battle", "|split|p1", "|win|Alice"], "Private protocol"),
    ],
)
def test_bad_replays_are_rejected_before_writing(tmp_path, lines, error):
    out = tmp_path / "replays.jsonl"
    battle = SimpleNamespace(won=True, lost=False, player_role="p1")
    with pytest.raises(RuntimeError, match=error):
        write_public_logs(out, {"room": battle}, {"room": lines})
    assert not out.exists()
