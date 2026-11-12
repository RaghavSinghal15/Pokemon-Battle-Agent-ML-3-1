"""Building the payoff table for the current turn."""

from poke_env.battle import Battle

from pokeagent.interfaces import BeliefState, PayoffTable


def build_payoff_table(
    battle: Battle, beliefs: BeliefState, class_probs: dict[str, float] | None
) -> PayoffTable:
    """One row per legal action for us, one column per plausible opponent action.

    TODO (role F):
    - Our actions: battle.available_moves and battle.available_switches.
    - Opponent actions: moves the opponent's active Pokémon probably has (from `beliefs`), plus
      one aggregated "switch" column.
    - Column probabilities: split each action class's probability (from the opponent model)
      across the columns in that class.
    - Cell value (version 1): a one-turn evaluation using poke-env's damage calculator
      (poke_env.calc.damage_calc_gen9.calculate_damage), e.g. expected HP fraction we remove
      minus HP fraction we lose, with a bonus for knock-outs. Account for who moves first.
    - Write down every simplification in docs/decisions.md; each one is a viva question.
    """
    raise NotImplementedError
