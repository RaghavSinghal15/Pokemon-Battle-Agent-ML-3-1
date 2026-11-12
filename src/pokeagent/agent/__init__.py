"""Role F (with E): the live agent and its explanation log.

Wires every component together inside a poke-env Player that plays on the local server:

    live battle -> parser.live -> beliefs -> features -> opponent model
                -> decision.payoff -> best response or LP -> move + explanation
"""
