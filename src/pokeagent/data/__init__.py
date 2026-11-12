"""Role B: data pipeline. Pokémon knowledge needed: none.

Turns the raw Hugging Face replay dataset into a clean, filtered table of battles and a fixed
train / validation / test split that everyone uses.

Produces:
    data/interim/battles.parquet   one row per battle: battle_id, upload_time, rating_p1,
                                   rating_p2, winner ("p1"/"p2"), log (the full battle log text)
    data/splits.json               {"train": [...], "val": [...], "test": [...]} battle ids
    reports/data_report.md         counts, rating distribution, game lengths, what was dropped
"""
