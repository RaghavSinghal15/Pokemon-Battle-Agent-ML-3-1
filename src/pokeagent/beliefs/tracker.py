"""Bayesian updates over the opponent's hidden sets."""

from pokeagent.interfaces import BeliefState, PokemonView, PublicState, SetBelief


def prior_belief(species: str) -> SetBelief:
    """The belief before anything about this Pokémon is revealed (from the set data)."""
    raise NotImplementedError


def update_belief(belief: SetBelief, seen: PokemonView) -> SetBelief:
    """Rule out roles that contradict what has been revealed, then rescale.

    Bayes' rule in its simplest form: P(role | evidence) is proportional to
    P(evidence | role) * P(role). If a revealed move is not in a role's move pool,
    P(evidence | role) = 0 and the role is eliminated. The surviving probabilities are divided
    by their sum so they add up to 1 again.

    TODO (role D): also update move, item, ability and tera probabilities, and write tests for
    the obvious cases (a move only one role has -> that role becomes probability 1).
    """
    raise NotImplementedError


def beliefs_for(state: PublicState) -> BeliefState:
    """Beliefs for every revealed opponent Pokémon in `state`."""
    return {p.species: update_belief(prior_belief(p.species), p) for p in state.opp.pokemon}
