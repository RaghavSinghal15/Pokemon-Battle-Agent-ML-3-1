"""Role D: belief tracking. Pokémon knowledge needed: high.

Keeps probabilities over each opponent Pokémon's hidden details (its role, moves, item,
ability and tera type) and updates them as the battle reveals information.

Sources of prior knowledge:
    - Showdown's data/random-battles/gen9/sets.json: for each species, its possible roles, each
      with a move pool (4 moves are picked from it), abilities and tera types. Items are chosen
      by Showdown's team generator code, not listed in this file.
    - The version 1 model gives each template equal weight and leaves hidden items unknown.
      Public item reveals still update the estimate of what is currently held.

Produces: a BeliefState (see interfaces.py) for any PublicState.
"""
