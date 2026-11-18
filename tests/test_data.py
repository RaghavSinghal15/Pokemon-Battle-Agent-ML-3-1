"""Tests for role B's data pipeline, on small hand-made battles (no download needed)."""

import json

from pokeagent.data.ingest import BATTLE_COLUMNS, battles_from_rows, summarise_log, to_id


def make_log(
    p1="Ash K.", p2="Misty", r1="1500", r2="1320", turns=3, winner="Ash K.", tie=False
) -> str:
    """A tiny but realistic battle log with the lines our code reads."""
    lines = [
        "|j|☆Ash K.",
        f"|player|p1|{p1}|red|{r1}",
        f"|player|p2|{p2}|misty|{r2}",
        "|gametype|singles",
        "|tier|[Gen 9] Random Battle",
        "|start",
    ]
    for t in range(1, turns + 1):
        lines += [f"|turn|{t}", "|move|p1a: Pikachu|Thunderbolt|p2a: Starmie"]
    if tie:
        lines.append("|tie")
    elif winner:
        lines.append(f"|win|{winner}")
    return "\n".join(lines)


def make_row(battle_id="gen9randombattle-1", formatid="gen9randombattle", **log_args) -> dict:
    return {
        "id": battle_id,
        "formatid": formatid,
        "uploadtime": 1_700_000_000,
        "log": make_log(**log_args),
    }


def test_to_id_matches_showdown_names():
    assert to_id("Ash K.") == "ashk"
    assert to_id("MISTY_99") == "misty99"


def test_summary_reads_ratings_winner_and_turns():
    s = summarise_log(make_log(turns=7, winner="Misty"))
    assert (s["rating_p1"], s["rating_p2"]) == (1500, 1320)
    assert s["winner"] == "p2"
    assert s["turns"] == 7
    assert s["tie"] is False


def test_winner_matched_even_if_written_differently():
    # Showdown compares names ignoring case, spaces and punctuation.
    assert summarise_log(make_log(winner="ash k"))["winner"] == "p1"


def test_unrated_unfinished_and_tied_games():
    unrated = summarise_log(make_log(r1="", r2=""))
    assert unrated["rating_p1"] is None and unrated["rating_p2"] is None
    unfinished = summarise_log(make_log(winner=None))
    assert unfinished["winner"] is None and unfinished["tie"] is False
    tied = summarise_log(make_log(tie=True))
    assert tied["winner"] is None and tied["tie"] is True


def test_reconnect_player_line_does_not_overwrite_name():
    # On reconnect Showdown sends "|player|p1|" with no name; the first named line must win.
    log = make_log() + "\n|player|p1|"
    s = summarise_log(log)
    assert s["rating_p1"] == 1500 and s["winner"] == "p1"


def test_battles_from_rows_keeps_only_our_format():
    rows = [
        make_row("gen9randombattle-1"),
        make_row("gen9randombattleblitz-2", formatid="gen9randombattleblitz"),
        make_row("gen8randombattle-3", formatid="gen8randombattle"),
    ]
    df = battles_from_rows(rows)
    assert list(df.columns) == BATTLE_COLUMNS
    assert list(df["battle_id"]) == ["gen9randombattle-1"]
    row = df.iloc[0]
    assert json.loads(row["ratings"]) == {"p1": 1500, "p2": 1320}
    assert row["winner"] == "p1" and row["turns"] == 3


def test_battles_from_rows_skips_missing_logs():
    row = make_row()
    row["log"] = None
    assert battles_from_rows([row]).empty


def test_download_streams_batches_and_respects_max_battles(tmp_path, monkeypatch):
    """Run download_replays on a fake local "dataset" file instead of Hugging Face."""
    import huggingface_hub
    import pandas as pd

    from pokeagent.data import ingest

    rows = [make_row(f"gen9randombattle-{i}") for i in range(5)]
    rows.append(make_row("gen9randombattleblitz-9", formatid="gen9randombattleblitz"))
    source = tmp_path / "part1.parquet"
    pd.DataFrame(rows).to_parquet(source)

    class LocalFileSystem:  # stands in for HfFileSystem: every remote path opens our file
        def open(self, path, mode):
            return open(source, mode)

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", LocalFileSystem)
    monkeypatch.setattr(ingest, "list_source_files", lambda revision: ["part1.parquet"])

    out = ingest.download_replays(tmp_path / "raw.parquet", max_battles=3)
    df = ingest.load_battles(out)
    assert list(df["battle_id"]) == [f"gen9randombattle-{i}" for i in range(3)]

    everything = ingest.load_battles(ingest.download_replays(tmp_path / "all.parquet"))
    assert len(everything) == 5  # the Blitz game is dropped
