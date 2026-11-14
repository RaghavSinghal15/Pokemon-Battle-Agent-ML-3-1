"""Read Gen 9 singles replays into public snapshots and observable choices."""

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from typing import cast

from poke_env.battle import Battle, Field, Pokemon
from poke_env.data import GenData, to_id_str

from pokeagent.interfaces import (
    Action,
    DecisionPoint,
    FieldView,
    Player,
    PokemonView,
    PublicState,
    SideView,
)

_GENERATION = 9
_TEAM_SIZE = 6
_PLAYERS: tuple[Player, Player] = ("p1", "p2")
_UNKNOWN_ITEM = GenData.UNKNOWN_ITEM
_LOGGER = logging.getLogger(__name__)
_SKIP = {
    "",
    "request",
    "showteam",
    "bigerror",
    "error",
    "t:",
    "c:",
    "raw",
    "uhtmlchange",
    "title",
    "win",
    "tie",
    "inactive",
    "inactiveoff",
    "challstr",
    "updateuser",
}


class ReplayParseError(ValueError):
    """A replay cannot be converted without risking an incorrect public state."""

    def __init__(self, battle_id: str, event: str):
        self.event = event
        super().__init__(f"{battle_id}: invalid or unsupported {event!r} event")


def _public_events(battle_id: str, lines: Iterable[str]) -> Iterator[list[str]]:
    """Discard the private half of Showdown's split events, when present."""
    source = iter(lines)
    for line in source:
        if line.startswith("|split|"):
            private = next(source, "")
            line = next(source, "")
            if (
                not private.startswith("|")
                or not line.startswith("|")
                or line.startswith("|split|")
            ):
                raise ReplayParseError(battle_id, "split")
        parts = line.rstrip("\r\n").split("|")
        if len(parts) > 1 and not parts[0]:
            yield parts


def _public_hp(pokemon: Pokemon) -> float:
    if not pokemon.max_hp or pokemon.fainted:
        return 0.0
    current, maximum = pokemon.current_hp, pokemon.max_hp
    percentage = (100 * current + maximum - 1) // maximum
    return (min(99, percentage) if current < maximum else 100) / 100


def _pokemon_view(pokemon: Pokemon, species: str) -> PokemonView:
    item = pokemon.item
    return PokemonView(
        species=species,
        level=pokemon.level,
        hp_fraction=_public_hp(pokemon),
        status=pokemon.status.name.lower() if pokemon.status and not pokemon.fainted else None,
        boosts={stat: value for stat, value in pokemon.boosts.items() if value},
        revealed_moves=sorted(pokemon.moves),
        revealed_item=item if item and item != _UNKNOWN_ITEM else None,
        item_removed=not item,
        revealed_ability=pokemon.ability,
        tera_type=pokemon.tera_type.name.lower() if pokemon.tera_type else None,
        terastallized=pokemon.is_terastallized,
        fainted=pokemon.fainted,
        active=bool(pokemon.active),
    )


class _Replay:
    """Track both public sides with a fixed orientation for replay and live input."""

    def __init__(self, battle_id: str, ratings: Mapping[str, int | None]):
        self.battle = Battle(battle_id, "p1", _LOGGER, gen=_GENERATION, save_replays=False)
        self.battle.player_role = "p1"
        self.names: dict[Player, str] = {}
        self.ratings = dict(ratings)
        self.sizes: dict[Player, int] = {}
        # Keep the object alongside its initial species so an identity cannot be recycled.
        self.species: dict[int, tuple[Pokemon, str]] = {}
        self.last: dict[Player, Action | None] = {}
        self.choices: dict[Player, Action | None] = {}
        self.tera: set[Player] = set()
        self.switch_window = False
        self.choice_window = False

    def snapshot(self) -> tuple[PublicState, PublicState]:
        battle = self.battle
        sides = []
        for player, team, conditions, tera_used in (
            ("p1", battle.team, battle.side_conditions, battle.used_tera),
            (
                "p2",
                battle.opponent_team,
                battle.opponent_side_conditions,
                battle.opponent_used_tera,
            ),
        ):
            sides.append(
                SideView(
                    player=cast(Player, player),
                    pokemon=[
                        _pokemon_view(mon, self.species[id(mon)][1])
                        for mon in team.values()
                        if id(mon) in self.species
                    ],
                    team_size=self.sizes.get(player, _TEAM_SIZE),
                    side_conditions={
                        to_id_str(condition.name): count for condition, count in conditions.items()
                    },
                    tera_used=tera_used,
                    rating=self.ratings.get(player),
                )
            )
        terrain = next((field for field in battle.fields if field.is_terrain), None)
        weather = next(iter(battle.weather), None)
        first = PublicState(
            battle.battle_tag,
            battle.turn,
            "p1",
            sides[0],
            sides[1],
            FieldView(
                weather=to_id_str(weather.name) if weather else None,
                terrain=to_id_str(terrain.name) if terrain else None,
                trick_room=Field.TRICK_ROOM in battle.fields,
            ),
            self.last.get("p1"),
            self.last.get("p2"),
        )
        second = replace(
            first,
            perspective="p2",
            me=first.opp,
            opp=first.me,
            my_last_action=first.opp_last_action,
            opp_last_action=first.my_last_action,
        )
        return first, second

    def _choice(self, event: list[str]) -> None:
        kind = event[1]
        if kind in {"upkeep", "win", "tie"}:
            self.choice_window = False
        if not self.choice_window or len(event) < 3 or event[2][:2] not in _PLAYERS:
            return
        side = cast(Player, event[2][:2])
        if kind == "-terastallize":
            self.tera.add(side)
        if kind in {"move", "cant", "faint"}:
            self.switch_window = False
        if side in self.choices:
            return
        if kind == "switch" and self.switch_window:
            self.choices[side] = Action("switch", to_id_str(event[3].split(",")[0]))
        elif kind == "move":
            sources = [tag[6:].strip() for tag in event[4:] if tag.startswith("[from]")]
            if not sources:
                self.choices[side] = Action("move", to_id_str(event[3]), side in self.tera)
            elif any(to_id_str(source) == "lockedmove" for source in sources):
                self.choices[side] = None
        elif kind in {"cant", "faint"} or (kind == "-damage" and "[from] confusion" in event[4:]):
            self.choices[side] = None

    def feed(self, event: list[str]) -> tuple[PublicState, PublicState] | None:
        kind = event[1]
        try:
            if kind == "player":
                side = cast(Player, event[2])
                if event[3] and side in _PLAYERS:
                    self.names[side] = event[3]
                    if len(event) > 5 and event[5].isdigit() and self.ratings.get(side) is None:
                        self.ratings[side] = int(event[5])
                return None
            self._choice(event)
            if kind in _SKIP:
                return None
            if kind == "teamsize":
                self.sizes[cast(Player, event[2])] = int(event[3])
            self.battle.parse_message(event)
            if kind in {"switch", "drag", "replace"}:
                mon = (
                    self.battle.active_pokemon
                    if event[2].startswith("p1")
                    else self.battle.opponent_active_pokemon
                )
                if mon is not None:
                    self.species.setdefault(id(mon), (mon, to_id_str(event[3].split(",")[0])))
            if kind == "turn":
                self.last, self.choices = self.choices, {}
                self.tera.clear()
                self.switch_window = self.choice_window = True
                return self.snapshot()
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            AssertionError,
            NotImplementedError,
        ) as exc:
            raise ReplayParseError(self.battle.battle_tag, kind) from exc
        return None


def _labelled(
    states: tuple[PublicState, PublicState], choices: dict[Player, Action | None]
) -> Iterator[tuple[PublicState, Action]]:
    for state in states:
        action = choices.get(state.perspective)
        if action is not None:
            yield state, action


def iter_public_states(
    battle_id: str, lines: Iterable[str], ratings: Mapping[str, int | None] | None = None
) -> Iterator[PublicState]:
    """Yield both perspectives at each turn boundary, including unlabelled turns."""
    replay = _Replay(battle_id, ratings or {})
    for event in _public_events(battle_id, lines):
        states = replay.feed(event)
        if states is not None:
            yield from states


def parse_battle(
    battle_id: str, log: str, winner: str | None, ratings: Mapping[str, int | None]
) -> list[DecisionPoint]:
    """Return observable choices paired with their start-of-turn public states."""
    replay = _Replay(battle_id, ratings)
    states: tuple[PublicState, PublicState] | None = None
    observed = []
    for event in _public_events(battle_id, log.splitlines()):
        kind = event[1]
        if kind == "turn" and states:
            observed.extend(_labelled(states, replay.choices))
        if kind == "win" and winner is None and len(event) > 2:
            winner = event[2]
        new_states = replay.feed(event)
        if new_states is not None:
            states = new_states
    if states:
        observed.extend(_labelled(states, replay.choices))
    winning_side = (
        winner
        if winner in _PLAYERS
        else next(
            (
                side
                for side, name in replay.names.items()
                if winner and to_id_str(name) == to_id_str(winner)
            ),
            None,
        )
    )
    if winner and winning_side is None:
        raise ReplayParseError(battle_id, "winner")
    return [
        DecisionPoint(state, action, state.perspective == winning_side)
        for state, action in observed
    ]
