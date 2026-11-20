"""Role B: data pipeline. Pokémon knowledge needed: none.

Turns the raw Hugging Face replay dataset into a clean, filtered table of battles and a fixed
train / validation / test split that everyone uses.

Produces:
    data/raw/gen9randombattle.parquet   every downloaded battle, one row each (see ingest.py
                                        for the columns)
    data/interim/battles.parquet   the filtered battles, same columns
    data/splits.json               {"train": [...], "val": [...], "test": [...]} battle ids
    reports/data_report.md         counts, rating distribution, game lengths, what was dropped
"""
