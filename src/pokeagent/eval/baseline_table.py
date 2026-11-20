"""Combine balanced baseline matchups into an auditable report, without choosing winners."""

from collections import defaultdict

from pokeagent.config import BATTLE_FORMAT
from pokeagent.eval.baselines import BASELINES
from pokeagent.eval.stats import win_rate_ci


def summarise_matchups(records: list[dict]) -> list[dict]:
    """Require opposite, equally sized orientations and consistent clean provenance.

    A partial batch cannot masquerade as a completed comparison. Rates use all games,
    including ties as non-wins for both bots; they are not half-credit scores.
    """
    if not records:
        raise ValueError("No matchup records")
    labels = {cls.__name__: name for name, cls in BASELINES.items()}
    groups = defaultdict(list)
    source, server = records[0]["source"], records[0]["server_source"]
    if source.get("tracked_changes") or not source.get("commit"):
        raise ValueError("Formal baselines require committed source without tracked changes")
    if not server or server.get("tracked_changes") or not server.get("commit"):
        raise ValueError("Formal baselines require a recorded clean server revision")
    for record in records:
        if record["source"] != source or record["server_source"] != server:
            raise ValueError("Source/server provenance differs between matchups")
        if record["format"] != BATTLE_FORMAT:
            raise ValueError("Baseline format differs from the project format")
        counts = [record[k] for k in ("games", "wins", "losses", "ties")]
        if (
            any(type(n) is not int or n < 0 for n in counts)
            or counts[0] == 0
            or sum(counts[1:]) != counts[0]
        ):
            raise ValueError("Invalid matchup counts")
        try:
            player, opponent = labels[record["player_class"]], labels[record["opponent_class"]]
        except KeyError as error:
            raise ValueError("Unknown baseline class") from error
        if player == opponent:
            raise ValueError("A baseline pair must contain two different bots")
        groups[tuple(sorted((player, opponent)))].append((player, record))
    summary = []
    for (player, opponent), games in sorted(groups.items()):
        if len(games) != 2 or {p for p, _ in games} != {player, opponent}:
            raise ValueError("Each pair needs exactly one matchup in each direction")
        if games[0][1]["games"] != games[1][1]["games"]:
            raise ValueError("Challenge directions must contain equal game counts")
        total = sum(record["games"] for _, record in games)
        wins = sum(record["wins"] if p == player else record["losses"] for p, record in games)
        losses = sum(record["losses"] if p == player else record["wins"] for p, record in games)
        ties = sum(record["ties"] for _, record in games)
        rate, low, high = win_rate_ci(wins, total)
        opp_rate, opp_low, opp_high = win_rate_ci(losses, total)
        summary.append(
            {
                "player": player,
                "opponent": opponent,
                "games": total,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": rate,
                "ci_lower": low,
                "ci_upper": high,
                "opponent_win_rate": opp_rate,
                "opponent_ci_lower": opp_low,
                "opponent_ci_upper": opp_high,
            }
        )
    return summary


def render_baseline_report(records: list[dict], record_links: list[str]) -> str:
    """Render the three fixed baseline comparisons and links to their six source records."""
    summary = summarise_matchups(records)
    if len(summary) != 3 or len(record_links) != len(records):
        raise ValueError("The full baseline report needs all three pairs and matching record links")
    source, server = records[0]["source"]["commit"], records[0]["server_source"]["commit"]
    lines = [
        "# Baseline bot comparisons (A3)",
        "",
        f"Format: `{BATTLE_FORMAT}`. Evaluator source: `{source}`. Showdown: `{server}`.",
        "",
        "Each pair uses equal numbers in both challenge directions. This is baseline-only",
        "evaluation; no agent, model or human-replay test set is evaluated here.",
        "",
        "| First bot | Second bot | Games | First wins | Second wins | Ties | "
        "First win rate (95% Wilson) | Second win rate (95% Wilson) |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary:
        first = f"{row['win_rate']:.1%} [{row['ci_lower']:.1%}, {row['ci_upper']:.1%}]"
        second = (
            f"{row['opponent_win_rate']:.1%} "
            f"[{row['opponent_ci_lower']:.1%}, {row['opponent_ci_upper']:.1%}]"
        )
        lines.append(
            f"| {row['player']} | {row['opponent']} | {row['games']} | {row['wins']} | "
            f"{row['losses']} | {row['ties']} | {first} | {second} |"
        )
    lines += [
        "",
        "Win rate means wins / all games. Ties are non-wins for both bots, not half-credit.",
        "Wilson intervals use z=1.96 and describe sampling uncertainty, not a guarantee.",
        "No claim of statistically significant differences or integrated-agent strength is made.",
        "",
        "Python/NumPy seeds, UTC timestamps, package versions and durations are in the records.",
        "The server's random teams are not seeded by the Python seed, so replaying this command",
        "will produce another random sample. Source flags describe tracked changes only.",
        "",
        "## Recorded matchups",
        "",
    ]
    for record, link in zip(records, record_links, strict=True):
        lines.append(
            f"- [{record['player_class']} vs {record['opponent_class']}, "
            f"{record['games']} games, seed {record['seed']}]({link})"
        )
    return "\n".join(lines) + "\n"
