"""Role D: belief tracking. Pokémon knowledge needed: high.

Keeps probabilities over each opponent Pokémon's hidden details (its role, moves, item,
ability and tera type) and updates them as the battle reveals information.

Sources of prior knowledge:
    - Showdown's data/random-battles/gen9/sets.json: for each species, its possible roles, each
      with a move pool (4 moves are picked from it), abilities and tera types. Items are chosen
      by Showdown's team generator code, not listed in this file.
    - The randbats stats (pkmn.github.io/randbats): how often each move, item, ability and tera
      type actually appears for each species and role in generated teams. Use these as priors.

Produces: a BeliefState (see interfaces.py) for any PublicState.
"""
