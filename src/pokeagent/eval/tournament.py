"""Running many games between two players on the local server."""

from poke_env.player import Player


async def run_matchup(player: Player, opponent: Player, n_games: int) -> dict:
    """Play `n_games` and return wins, losses, ties and the confidence interval.

    TODO (role A): build on scripts/play_baselines.py. Save each matchup's result to
    reports/results/ together with the git commit hash and the seed, so every number in the
    report can be traced back to the code that produced it.
    """
    raise NotImplementedError
