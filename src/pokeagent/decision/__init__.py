"""Role F: decision engine. Pokémon knowledge needed: medium (judging whether payoffs make sense).

Builds a payoff table of (our action, their action) outcomes and picks a move with one of two
strategies:
    best response   maximise the expected payoff against the opponent model's probabilities
    LP (minimax)    the mixed strategy with the best GUARANTEED payoff against any opponent,
                    found with linear programming (scipy.optimize.linprog)
"""
