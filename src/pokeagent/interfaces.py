"""Shared data structures: the contract between modules.

READ THIS FILE FIRST. Every module either produces or consumes these objects:

    replay logs --(parser)--> PublicState + Action per turn = DecisionPoint
    PublicState --(beliefs)--> BeliefState
    PublicState + BeliefState --(features)--> feature row (dict of numbers)
    feature row --(models)--> probabilities over the opponent's next action
    everything above --(decision)--> PayoffTable --> Decision (with an explanation)

Because everyone codes against these definitions, people can work in parallel: if your input
does not exist yet, build a few fake objects by hand and develop against those.

Changing anything in this file affects every role. Open a pull request, tag the owners of
every module that uses the field, and get their approval before merging (see CONTRIBUTING.md).

A note for people new to Pokémon: each field has a comment saying what it means in the game.
docs/03-pokemon-primer.md explains the concepts in more detail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Player = Literal["p1", "p2"]

# ---------------------------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------------------------


@dataclass
class Action:
    """One choice a player makes on a turn.

    In a singles battle a player either uses one of its active Pokémon's moves, or switches the
    active Pokémon out for a teammate. When using a move, they may also terastallize (change the
    Pokémon's type, allowed once per battle).
    """

    kind: Literal["move", "switch"]
    name: str  # move id ("earthquake") for a move, species id ("garchomp") for a switch
    tera: bool = False  # True if the player also terastallized this turn


# Version 1 label set for the opponent model: instead of predicting the exact move (every
# Pokémon has different moves), we predict the TYPE of action. Owned by roles D and E jointly;
# changing it is an interface change.
#   attack        a move that deals damage
#   non_damaging  a move that does not deal damage (stat boosts, healing, status, hazards...)
#   switch        swapping the active Pokémon out
ACTION_CLASSES: tuple[str, ...] = ("attack", "non_damaging", "switch")


# ---------------------------------------------------------------------------------------------
# What a spectator can see
# ---------------------------------------------------------------------------------------------


@dataclass
class PokemonView:
    """Everything publicly known about one Pokémon at a moment in the battle.

    "Revealed" means it has appeared in the battle log so far. Anything not revealed yet is
    unknown, even if it is revealed later in the game (using it early would be leakage).
    """

    species: str  # normalised id, e.g. "garchomp"
    level: int  # Random Battles balance weak and strong Pokémon by giving them different levels
    hp_fraction: float = 1.0  # remaining health from 0.0 (fainted) to 1.0 (full)
    status: str | None = None  # brn burn, par paralysis, slp sleep, frz freeze, psn poison,
    # tox badly poisoned; None = healthy
    boosts: dict[str, int] = field(default_factory=dict)  # stat changes from -6 to +6, e.g.
    # {"atk": 2} after Swords Dance; keys: atk, def, spa, spd, spe, accuracy, evasion
    revealed_moves: list[str] = field(default_factory=list)  # move ids seen so far
    revealed_item: str | None = None  # held item, once revealed
    item_removed: bool = False  # item was used up or knocked off, so it now holds nothing
    revealed_ability: str | None = None  # passive ability, once revealed
    tera_type: str | None = None  # the type it changed into, once it terastallizes
    terastallized: bool = False
    fainted: bool = False
    active: bool = False  # True if this is the Pokémon currently on the field


@dataclass
class SideView:
    """One player's side of the field, as a spectator sees it."""

    player: Player
    pokemon: list[PokemonView] = field(default_factory=list)  # revealed so far, in order seen
    team_size: int = 6  # every Random Battle team has 6 Pokémon
    side_conditions: dict[str, int] = field(default_factory=dict)  # hazards and screens on
    # this side, e.g. {"stealthrock": 1, "spikes": 2, "reflect": 1}; value = layers
    tera_used: bool = False  # each player may terastallize once per battle
    rating: int | None = None  # the player's ladder rating, if known

    @property
    def active(self) -> PokemonView | None:
        return next((p for p in self.pokemon if p.active), None)

    @property
    def n_unrevealed(self) -> int:
        """How many team members have not appeared yet."""
        return self.team_size - len(self.pokemon)

    @property
    def n_remaining(self) -> int:
        """How many team members have not fainted (unrevealed ones count as alive)."""
        return self.team_size - sum(p.fainted for p in self.pokemon)


@dataclass
class FieldView:
    """Effects on the whole field, shared by both sides."""

    weather: str | None = None  # e.g. "RainDance", "SunnyDay", "Sandstorm", "Snow"
    terrain: str | None = None  # e.g. "Electric Terrain", "Grassy Terrain"
    trick_room: bool = False  # while active, slower Pokémon move first


@dataclass
class PublicState:
    """The battle as a spectator sees it at the START of a turn, before either player chooses.

    `perspective` is the player whose decision we are looking at; `me` is their side and `opp`
    is the other side. Both sides contain only public (revealed) information, even `me`.

    Why public information only, even for our own side? Replays are recorded from the
    spectator's view, so the training data never contains a player's unrevealed moves. If live
    play used more information than training did, the model would see inputs at play time that
    it never saw in training. Using the same public view in both places keeps them consistent.
    The decision engine may still use our full team when playing live (it is not a model input).
    """

    battle_id: str
    turn: int
    perspective: Player
    me: SideView
    opp: SideView
    field: FieldView = field(default_factory=FieldView)
    my_last_action: Action | None = None
    opp_last_action: Action | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PublicState:
        def side(s: dict) -> SideView:
            return SideView(
                player=s["player"],
                pokemon=[PokemonView(**p) for p in s["pokemon"]],
                team_size=s["team_size"],
                side_conditions=s["side_conditions"],
                tera_used=s["tera_used"],
                rating=s["rating"],
            )

        def action(a: dict | None) -> Action | None:
            return Action(**a) if a else None

        return cls(
            battle_id=d["battle_id"],
            turn=d["turn"],
            perspective=d["perspective"],
            me=side(d["me"]),
            opp=side(d["opp"]),
            field=FieldView(**d["field"]),
            my_last_action=action(d["my_last_action"]),
            opp_last_action=action(d["opp_last_action"]),
        )


@dataclass
class DecisionPoint:
    """One training example: the state before a turn, and what the player actually did.

    The parser produces two decision points per turn (one from each player's perspective).
    Forced switches (choosing a replacement after a Pokémon faints) are not simultaneous
    choices and are excluded in version 1.
    """

    state: PublicState
    action: Action  # what `state.perspective` actually chose this turn
    won: bool  # did `state.perspective` win the battle in the end? A LABEL only, never a feature


# ---------------------------------------------------------------------------------------------
# Beliefs about hidden information
# ---------------------------------------------------------------------------------------------


@dataclass
class SetBelief:
    """Our probabilities about one opponent Pokémon's hidden details.

    In Random Battles each species has a small number of possible "roles" (usually 1 to 3),
    each with its own pool of moves, abilities and tera types. Revealed information rules roles
    out, and the probabilities of the remaining roles are rescaled (Bayes' rule).

    All dictionaries map a name to a probability. `role_probs` sums to 1. `move_probs` gives,
    for each move, the probability that this Pokémon has it (these do NOT sum to 1, since a
    Pokémon has four moves).
    """

    species: str
    role_probs: dict[str, float] = field(default_factory=dict)
    move_probs: dict[str, float] = field(default_factory=dict)
    item_probs: dict[str, float] = field(default_factory=dict)
    ability_probs: dict[str, float] = field(default_factory=dict)
    tera_probs: dict[str, float] = field(default_factory=dict)


# species id -> belief, for every opponent Pokémon revealed so far
BeliefState = dict[str, SetBelief]

# A feature row: feature name -> number. Every row from features.build must have the same keys
# in the same order; features.FEATURE_NAMES is the list.
FeatureRow = dict[str, float]


# ---------------------------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------------------------


@dataclass
class PayoffTable:
    """How good each (our action, their action) combination is for us.

    values[i][j] is the payoff for us if we choose our_actions[i] and the opponent chooses
    opp_actions[j]. Higher is better for us. opp_probs[j] is the opponent model's probability
    that they choose opp_actions[j] (None if no model is used).
    """

    our_actions: list[Action]
    opp_actions: list[Action]
    values: list[list[float]]
    opp_probs: list[float] | None = None


@dataclass
class Decision:
    """What the agent chose on a turn, and why."""

    chosen: Action
    strategy: Literal["best_response", "lp", "baseline"]
    mixed_strategy: dict[str, float] = field(default_factory=dict)  # action label -> probability
    explanation: str = ""  # human-readable reasoning printed in the explanation log
