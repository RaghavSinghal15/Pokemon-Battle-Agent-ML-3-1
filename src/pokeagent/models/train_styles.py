"""E7 (#80): cluster players by play style, plot them, and test a style feature.

    python -m pokeagent.models.train_styles

Fits everything on training games only. Writes reports/results/styles.json and
reports/figures/styles_pca.png (clusters and coordinates only, never usernames).
"""

import inspect
import json

import pandas as pd

from pokeagent.config import BATTLES_PATH, DECISION_POINTS_PATH, REPORTS_DIR, SEED, SPLITS_PATH
from pokeagent.features.build import FEATURE_NAMES
from pokeagent.models.opponent import (
    LABEL,
    LGBM_EARLY_STOP,
    LGBM_LEARNING_RATE,
    LGBM_MAX_TREES,
    WITH_HIDDEN,
    evaluate_all_turns,
    fold_hidden,
    load_split,
)
from pokeagent.models.styles import (
    STYLE_FEATURES,
    STYLE_STATS,
    cluster,
    game_profiles,
    player_ids,
    player_profiles,
    style_features,
)

REPORT_PATH = REPORTS_DIR / "results" / "styles.json"
FIGURE_PATH = REPORTS_DIR / "figures" / "styles_pca.png"
MIN_GAMES_TO_CLUSTER = 5
LEAVES = 63  # one fixed tree size for both runs, so only the features differ


class _Columns:
    """A LightGBM model on a chosen list of columns, answering like OpponentModel."""

    def __init__(self, name, booster, columns):
        self.name, self.booster, self.columns = name, booster, list(columns)

    def predict_proba_table(self, table):
        return fold_hidden(self.booster.predict_proba(table[self.columns].astype(float)))


def _fit(train, val, columns, name):
    import lightgbm as lgb

    classes = {c: i for i, c in enumerate(WITH_HIDDEN)}
    booster = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=LGBM_MAX_TREES,
        learning_rate=LGBM_LEARNING_RATE,
        num_leaves=LEAVES,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    x_val, y_val = val[columns].astype(float), val[LABEL].map(classes)
    if "eval_X" in inspect.signature(booster.fit).parameters:
        validation = {"eval_X": (x_val,), "eval_y": (y_val,)}
    else:
        validation = {"eval_set": [(x_val, y_val)]}
    booster.fit(
        train[columns].astype(float),
        train[LABEL].map(classes),
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(LGBM_EARLY_STOP, verbose=False)],
        **validation,
    )
    return _Columns(name, booster, columns)


def plot(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy, labels = result["pca_coordinates"], result["labels"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for k in range(result["chosen_k"]):
        points = xy[labels == k]
        centre = result["centres"][k]
        name = (
            f"style {k + 1}: attack {centre['attack']:.0%}, switch {centre['switch']:.0%},"
            f" other status {centre['other_status']:.0%}"
        )
        ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.5, label=name)
    explained = result["pca_explained"]
    ax.set_xlabel(f"PCA 1 ({explained[0]:.0%} of variance)")
    ax.set_ylabel(f"PCA 2 ({explained[1]:.0%} of variance)")
    ax.set_title("Players by play style (training games, k-means)")
    ax.legend(fontsize=7)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(clusters_only: bool = False) -> None:
    """clusters_only=True redoes the clustering and keeps the saved style-feature test."""
    battles = pd.read_parquet(BATTLES_PATH, columns=["battle_id", "log"])
    names = player_ids(battles)
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    split_of = {b: name for name, ids in splits.items() for b in ids}
    points = pd.read_parquet(DECISION_POINTS_PATH, columns=["battle_id", "perspective", "action"])
    profiles = game_profiles(points)
    profiles["player"] = [
        names.get(k) for k in zip(profiles.battle_id, profiles.perspective, strict=True)
    ]
    profiles["split"] = profiles["battle_id"].map(split_of)
    profiles = profiles.dropna(subset=["player"])
    train_profiles = profiles[profiles["split"] == "train"]
    players = player_profiles(train_profiles, min_games=MIN_GAMES_TO_CLUSTER)
    print(f"{len(players):,} players with {MIN_GAMES_TO_CLUSTER}+ training games")
    result = cluster(players)
    plot(result)

    if clusters_only:
        saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        comparison, known_share = saved["style_feature_test"], saved["val_rows_with_known_style"]
    else:
        comparison, known_share = _feature_test(names, train_profiles)
    report = {
        "players_clustered": len(players),
        "style_stats": list(STYLE_STATS),
        "profile_means": players[list(STYLE_STATS)].mean().to_dict(),
        **{k: v for k, v in result.items() if k not in ("labels", "pca_coordinates")},
        "val_rows_with_known_style": known_share,
        "style_feature_test": comparison,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    print(f"silhouette by k: { {k: round(v, 3) for k, v in result['k_silhouette'].items()} }")
    print(f"GMM BIC best k: {result['gmm_best_k']}; k-means chosen k: {result['chosen_k']}")
    for centre, size in zip(result["centres"], result["sizes"], strict=True):
        print(f"  {size:5} players: " + ", ".join(f"{k} {v:.0%}" for k, v in centre.items()))
    print(f"PCA explains {sum(result['pca_explained']):.0%} in 2-D")
    print(f"validation rows whose player has a known style: {known_share:.1%}")
    for name, r in comparison.items():
        print(
            f"  {name:28} all turns {r['all_turns_log_loss']:.4f}"
            f"  known players {r['known_players']:.4f}  trees {r['trees']}"
        )
    print(f"Wrote {REPORT_PATH} and {FIGURE_PATH}")


def _feature_test(names, train_profiles):
    """Does knowing the style help the opponent model? Same LightGBM, with and without it."""
    tables = {}
    for split in ("train", "val"):
        table = load_split(split, with_hidden=True)
        table["player"] = [
            names.get(k) for k in zip(table.battle_id, table.perspective, strict=True)
        ]
        tables[split] = pd.concat([table, style_features(table, train_profiles)], axis=1)
    train, val = tables["train"], tables["val"]
    base = _fit(train, val, list(FEATURE_NAMES), "LightGBM, 36 features")
    styled = _fit(train, val, [*FEATURE_NAMES, *STYLE_FEATURES], "LightGBM + style")
    known = val["style_known"] == 1
    comparison = {}
    for model in (base, styled):
        comparison[model.name] = {
            "all_turns_log_loss": evaluate_all_turns(model, val)["all_turns_log_loss"],
            "known_players": evaluate_all_turns(model, val[known])["all_turns_log_loss"],
            "trees": int(model.booster.best_iteration_),
        }
    return comparison, float(known.mean())


if __name__ == "__main__":
    import sys

    main(clusters_only="--clusters-only" in sys.argv)
