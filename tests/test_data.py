"""Tests for role B's data pipeline, on small hand-made battles (no download needed)."""

import json

import pandas as pd

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


def test_download_keeps_the_newest_battles(tmp_path, monkeypatch):
    """With a limit, the newest battles are kept even if they sit at the end of the file."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pokeagent.data import ingest

    rows = []
    for i in range(10):
        row = make_row(f"gen9randombattle-{i}")
        row["uploadtime"] = 1_700_000_000 + 100 * i  # later rows are newer
        rows.append(row)
    newest_blitz = make_row("blitz-99", formatid="gen9randombattleblitz")
    newest_blitz["uploadtime"] = 1_800_000_000  # newest of all, but not our format
    rows.append(newest_blitz)
    source = tmp_path / "part1.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=3)  # several blocks

    class LocalFileSystem:
        def open(self, path, mode):
            return open(source, mode)

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", LocalFileSystem)
    monkeypatch.setattr(ingest, "list_source_files", lambda revision: ["part1.parquet"])

    df = ingest.load_battles(ingest.download_replays(tmp_path / "raw.parquet", max_battles=4))
    assert sorted(df["battle_id"]) == [f"gen9randombattle-{i}" for i in (6, 7, 8, 9)]


def test_only_standard_gen9_files_are_listed(monkeypatch):
    import huggingface_hub

    from pokeagent.data import ingest

    names = [
        "README.md",
        "[Gen 9] RANDOMBATTLE (BLITZ).parquet",
        "[Gen 9] RANDOMBATTLE_part10.parquet",
        "[Gen 9] RANDOMBATTLE_part2.parquet",
        "[Gen 9] RANDOMBATTLE_part1.parquet",
        "[Gen 9] RANDOM DOUBLES BATTLE.parquet",
        "[Gen 8] RANDOMBATTLE_part1.parquet",
    ]

    class FakeApi:
        def list_repo_files(self, repo, repo_type, revision):
            return names

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert ingest.list_source_files() == [
        "[Gen 9] RANDOMBATTLE_part1.parquet",
        "[Gen 9] RANDOMBATTLE_part2.parquet",
        "[Gen 9] RANDOMBATTLE_part10.parquet",
    ]


# ---------------------------------------------------------------------------------------------
# B3: filtering
# ---------------------------------------------------------------------------------------------


def battle_table(rows: list[tuple]) -> pd.DataFrame:
    """Hand-made battle table: (battle_id, rating_p1, rating_p2, winner, turns) per row."""
    df = pd.DataFrame(rows, columns=["battle_id", "rating_p1", "rating_p2", "winner", "turns"])
    return df.astype({"rating_p1": "Int64", "rating_p2": "Int64", "winner": "string"})


def test_each_filter_drops_what_it_claims():
    from pokeagent.data.filter import filter_steps

    df = battle_table(
        [
            ("a", 1500, 1500, "p1", 20),  # kept
            ("a", 1500, 1500, "p1", 20),  # duplicate id
            ("b", 1500, 1500, None, 20),  # no winner (tie or unfinished)
            ("c", 1500, 1500, "p2", 1),  # too short
            ("d", 1500, 1100, "p1", 20),  # weaker player below the threshold
            ("e", 1500, None, "p2", 20),  # one player unrated
            ("f", 1300, 1300, "p2", 20),  # exactly at the threshold: kept
        ]
    )
    kept, counts = filter_steps(df, min_rating=1300, min_turns=2)
    assert sorted(kept["battle_id"]) == ["a", "f"]
    assert [n for _, n in counts] == [7, 6, 5, 4, 2]


def test_chosen_threshold_keeps_the_requested_fraction():
    from pokeagent.data.filter import choose_min_rating, filter_battles

    # 100 usable games whose lower-rated player is rated 1000, 1010, ..., 1990.
    df = battle_table([(str(i), 1000 + 10 * i, 2500, "p1", 10) for i in range(100)])
    threshold = choose_min_rating(df, keep_fraction=0.35, min_turns=2)
    kept = filter_battles(df, threshold, min_turns=2)
    assert 34 <= len(kept) <= 36
    assert kept["rating_p1"].min() >= threshold


def test_threshold_ignores_unusable_and_unrated_games():
    from pokeagent.data.filter import choose_min_rating

    usable = [(str(i), 1500, 1500, "p1", 10) for i in range(10)]
    junk = [("x", 1000, 1000, None, 10), ("y", None, 1000, "p1", 10), ("z", 1000, 1000, "p1", 1)]
    assert choose_min_rating(battle_table(usable + junk), keep_fraction=0.5) == 1500


def test_filter_battle_file_copies_kept_rows_once(tmp_path):
    from pokeagent.data.filter import filter_battle_file
    from pokeagent.data.ingest import battles_from_rows, load_battles

    rows = [make_row("g-1"), make_row("g-1"), make_row("g-2", r2="900"), make_row("g-3")]
    for row in rows:
        row["formatid"] = "gen9randombattle"
    raw = tmp_path / "raw.parquet"
    battles_from_rows(rows).to_parquet(raw)

    result = filter_battle_file(raw, tmp_path / "battles.parquet", min_rating=1300)
    kept = load_battles(tmp_path / "battles.parquet")
    assert sorted(kept["battle_id"]) == ["g-1", "g-3"]
    assert kept["log"].str.contains("|win|").all()  # logs are copied, not just ids
    assert result["counts"][-1][1] == 2 and result["keep_fraction"] is None
