"""The per-turn explanation log: why the agent chose what it chose."""

from pokeagent.interfaces import BeliefState, Decision, PayoffTable, PublicState


def format_turn(
    state: PublicState,
    beliefs: BeliefState,
    class_probs: dict[str, float],
    table: PayoffTable,
    decision: Decision,
) -> str:
    """Return a short, readable explanation of one turn, for example:

        Turn 7  Gholdengo (78%) vs Garchomp (100%)
        Beliefs: Garchomp is Setup Sweeper 82%, Fast Support 18%; Choice Scarf 0%
        Opponent prediction: attack 61%, switch 27%, non_damaging 12%
        Payoffs: Make It Rain +0.42 | Shadow Ball +0.18 | switch to Corviknight +0.05
        Chose Make It Rain (best response, expected payoff +0.42)

    TODO (roles F and E).
    """
    raise NotImplementedError
