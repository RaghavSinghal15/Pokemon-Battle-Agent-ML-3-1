"""Role A: infrastructure and evaluation. Pokémon knowledge needed: none.

Runs games between our agent and baseline bots on the local Showdown server and reports win
rates with confidence intervals, plus the ablation studies (turning one component off at a
time to measure what it contributes).

Produces:
    reports/results/tournament_<date>.csv   one row per matchup: players, games, wins, CI
"""
