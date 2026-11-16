"""Connect the public parser, belief tracker and feature builder for model decisions."""

from dataclasses import replace

from poke_env.data import GenData, to_id_str

from pokeagent.beliefs.tracker import beliefs_for, prior_belief, update_belief
from pokeagent.config import RANDOM_BATTLE_GENERATION
from pokeagent.interfaces import BeliefState, PublicState


def public_beliefs(state: PublicState) -> BeliefState:
    """Condition on public observations, including a temporary cosmetic-form adapter."""
    result = beliefs_for(state)
    dex = GenData.from_gen(RANDOM_BATTLE_GENERATION)
    for seen in state.opp.pokemon:
        if result[seen.species].move_probs:
            continue
        base = to_id_str(dex.pokedex.get(seen.species, {}).get("baseSpecies", ""))
        if base and base != seen.species:
            # D's pending cosmetic-form fix is not duplicated in its owned source files.
            updated = update_belief(prior_belief(base), replace(seen, species=base))
            result[seen.species] = replace(updated, species=seen.species)
    return result
