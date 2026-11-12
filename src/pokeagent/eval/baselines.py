"""The baseline bots we compare against (all provided by poke-env).

RandomPlayer            picks any legal action at random
MaxBasePowerPlayer      always uses the move with the highest listed power
SimpleHeuristicsPlayer  a hand-written rule-based bot; the strongest of the three
"""

from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

BASELINES = {
    "random": RandomPlayer,
    "max_power": MaxBasePowerPlayer,
    "heuristics": SimpleHeuristicsPlayer,
}
