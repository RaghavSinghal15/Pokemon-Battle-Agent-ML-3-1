"""Opponent play styles (E7, #80): describe players by how they play, cluster them, and test
whether knowing a player's style helps predict their next action.

1. Game profile: for one player in one game, the share of their decisions that were switches,
   attacks, setup moves (raise their own stats), hazards, other status moves and Protect.
2. Player profile: the decision-weighted average over that player's TRAINING games.
3. Clusters: k-means (each player goes to the nearest of k centres) and a Gaussian mixture
   (each cluster is a bell-shaped blob with its own spread; players get soft memberships),
   on standardised profiles, shown in 2-D with PCA (the two directions of largest variance).
4. Style feature: for a decision point, the player's profile from their OTHER training games
   (never the current game, so the label cannot leak in). Players with fewer than
   MIN_OTHER_GAMES other games get the average profile and a "style known" flag of 0.

Usernames are read from the local logs to link a player's games and never leave data/.
"""

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from poke_env.data import GenData, to_id_str

from pokeagent.config import RANDOM_BATTLE_GENERATION
from pokeagent.features.build import _move_signals
from pokeagent.models.labels import move_class

STYLE_STATS = ("switch", "attack", "setup", "hazard", "other_status", "protect")
STYLE_FEATURES = tuple(f"style_{name}" for name in STYLE_STATS) + ("style_known",)
# The six shares add up to 1, so one is redundant (attack = 1 - the rest). Clustering on all
# six would put the data on a flat 5-D plane, where a full-covariance Gaussian mixture has
# near-singular covariances and a misleading BIC. So clustering leaves attack out.
CLUSTER_STATS = tuple(name for name in STYLE_STATS if name != "attack")
MIN_OTHER_GAMES = 2
_PLAYER = re.compile(r"^\|player\|(p[12])\|([^|]+)\|", re.M)
_HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}


def player_ids(battles: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(battle_id, "p1"/"p2") -> normalised username, read from each log's |player| lines."""
    names = {}
    for battle_id, log in zip(battles["battle_id"], battles["log"], strict=True):
        for match in _PLAYER.finditer(log):
            names[(battle_id, match.group(1))] = to_id_str(match.group(2))
    return names


def action_kind(action: dict) -> str:
    """Which style statistic one action counts towards."""
    if action["kind"] == "switch":
        return "switch"
    move = action["name"]
    entry = GenData.from_gen(RANDOM_BATTLE_GENERATION).moves.get(move, {})
    if entry.get("stallingMove") or move in {"protect", "detect", "kingsshield", "spikyshield"}:
        return "protect"
    if move_class(move) == "attack":
        return "attack"
    if _move_signals(move)[0]:
        return "setup"
    if move in _HAZARDS:
        return "hazard"
    return "other_status"


def game_profiles(points: pd.DataFrame) -> pd.DataFrame:
    """One row per (battle_id, perspective): decision count and the share of each kind."""
    counts = defaultdict(lambda: dict.fromkeys(STYLE_STATS, 0))
    for battle_id, perspective, action in zip(
        points["battle_id"], points["perspective"], points["action"], strict=True
    ):
        if isinstance(action, str):
            action = json.loads(action)
        try:
            counts[(battle_id, perspective)][action_kind(action)] += 1
        except ValueError:
            continue  # an unknown move name: leave it out of the profile
    rows = []
    for (battle_id, perspective), c in counts.items():
        total = sum(c.values())
        if total:
            shares = {name: c[name] / total for name in STYLE_STATS}
            rows.append({"battle_id": battle_id, "perspective": perspective, "n": total, **shares})
    return pd.DataFrame(rows)


def _weighted(group: pd.DataFrame) -> pd.Series:
    weights = group["n"].to_numpy(dtype=float)
    shares = group[list(STYLE_STATS)].to_numpy(dtype=float)
    return pd.Series(weights @ shares / weights.sum(), index=list(STYLE_STATS))


def player_profiles(profiles: pd.DataFrame, min_games: int = 1) -> pd.DataFrame:
    """Decision-weighted average profile per player (index: player id), with game counts."""
    grouped = profiles.groupby("player")
    result = grouped[["n", *STYLE_STATS]].apply(_weighted)
    result["games"] = grouped.size()
    result["decisions"] = grouped["n"].sum()
    return result[result["games"] >= min_games]


def style_features(rows: pd.DataFrame, train_profiles: pd.DataFrame) -> pd.DataFrame:
    """STYLE_FEATURES for each row (needs battle_id and player): the player's profile from
    their training games other than this battle (leave-one-out, so a training row never sees
    its own game's actions). Players with too few other games get the training average.
    """
    counts = train_profiles[list(STYLE_STATS)].mul(train_profiles["n"], axis=0)
    counts = counts.assign(player=train_profiles["player"], n=train_profiles["n"], games=1)
    totals = counts.groupby("player").sum()
    # Summed per (battle, player): a few battles have the same player on both sides, and the
    # whole battle is left out for that player.
    own = counts.assign(battle_id=train_profiles["battle_id"]).groupby(["battle_id", "player"])
    own = own.sum()
    keys = pd.MultiIndex.from_arrays([rows["battle_id"], rows["player"]])
    total = totals.reindex(rows["player"]).fillna(0).to_numpy()
    this_game = own.reindex(keys).fillna(0).to_numpy()  # zero unless the row is a training game
    other = total - this_game  # columns: STYLE_STATS..., n, games
    n_other, games_other = other[:, len(STYLE_STATS)], other[:, len(STYLE_STATS) + 1]
    known = games_other >= MIN_OTHER_GAMES
    average = _weighted(train_profiles).to_numpy()
    shares = np.where(
        known[:, None],
        other[:, : len(STYLE_STATS)] / np.maximum(n_other, 1)[:, None],
        average[None, :],
    )
    result = np.column_stack([shares, known.astype(float)])
    return pd.DataFrame(result, columns=list(STYLE_FEATURES), index=rows.index)


def cluster(players: pd.DataFrame, ks=range(2, 9), seed: int = 0) -> dict:
    """k-means (silhouette per k) and Gaussian mixtures (BIC per k) on standardised profiles,
    plus a 2-D PCA projection. Returns scores, the chosen k, centres in original units and the
    PCA coordinates (no player names)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    x_raw = players[list(CLUSTER_STATS)].to_numpy(dtype=float)
    scaler = StandardScaler().fit(x_raw)
    x = scaler.transform(x_raw)
    silhouettes, bics, kmeans_models = {}, {}, {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
        kmeans_models[k] = km
        silhouettes[k] = float(silhouette_score(x, km.labels_))
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed).fit(x)
        bics[k] = float(gm.bic(x))
    best_k = max(silhouettes, key=silhouettes.get)
    km = kmeans_models[best_k]
    centres = []
    for centre in scaler.inverse_transform(km.cluster_centers_):
        shares = dict(zip(CLUSTER_STATS, map(float, centre), strict=True))
        centres.append({"attack": 1 - sum(shares.values()), **shares})
    pca = PCA(n_components=2, random_state=seed).fit(x)
    return {
        "k_silhouette": silhouettes,
        "gmm_bic": bics,
        "chosen_k": best_k,
        "gmm_best_k": min(bics, key=bics.get),
        "centres": centres,
        "sizes": np.bincount(km.labels_).tolist(),
        "labels": km.labels_,
        "pca_coordinates": pca.transform(x),
        "pca_explained": pca.explained_variance_ratio_.tolist(),
        "pca_axes": [dict(zip(CLUSTER_STATS, map(float, a), strict=True)) for a in pca.components_],
    }
