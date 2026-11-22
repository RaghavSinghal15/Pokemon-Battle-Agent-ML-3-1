"""Puzzle positions (V3, #73): hand-built turns where a good player agrees on the right move.

Win rates say whether the agent got better; puzzles say which decisions changed. Each puzzle
is one fake battle (our active Pokemon and reserves, their active Pokemon with a fully known
set) plus the accepted answer(s). The agent sees the opponent's moves through its beliefs, as
if they had all been revealed, so each puzzle tests the decision engine, not the beliefs.

    python scripts/run_puzzles.py
"""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from poke_env.battle import Battle, Move, Pokemon, Status

from pokeagent.config import SEED
from pokeagent.decision.damage import random_battle_stats
from pokeagent.decision.payoff import PayoffApproximationWarning, build_payoff_table
from pokeagent.decision.strategies import best_response, minimax_lp
from pokeagent.interfaces import ACTION_CLASSES, BeliefState, SetBelief

LIKELY_ATTACK = {"attack": 0.7, "non_damaging": 0.15, "switch": 0.15}
BLISSEY_LIKELY = {"attack": 0.7, "non_damaging": 0.25, "switch": 0.05}  # Toss, sometimes heals


@dataclass(frozen=True)
class Mon:
    """One Pokemon in a puzzle. hp is a fraction of max HP; level as in Random Battles."""

    species: str
    moves: tuple[str, ...] = ()
    level: int = 100
    hp: float = 1.0
    status: str | None = None  # "par", "brn", "tox" ...
    boosts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Puzzle:
    name: str
    idea: str  # the principle a good player would apply, in one sentence
    ours: Mon
    theirs: Mon
    accepted: frozenset[str]  # move ids or the species to switch to
    our_reserves: tuple[Mon, ...] = ()
    their_team_size: int = 6  # how many they have left in total, active included
    turn: int = 1
    # What a strong player would expect them to do (attack / non_damaging / switch). Part of
    # the puzzle, signed off with the answer: a flat 1/3 forecast would say Blissey uses Seismic
    # Toss only a third of the time, which nobody would assume.
    forecast: dict = field(default_factory=lambda: dict(LIKELY_ATTACK))


def _hp_text(fraction: float) -> str:
    return f"{round(fraction * 100)}/100"


def _apply(mon: Pokemon, spec: Mon) -> None:
    if spec.status:
        mon._status = Status[spec.status.upper()]
    for stat, value in spec.boosts.items():
        mon._boosts[stat] = value


def build(puzzle: Puzzle) -> tuple[Battle, BeliefState]:
    """The puzzle as a poke-env battle (as a live game would supply it) and our beliefs."""
    battle = Battle(f"battle-puzzle-{puzzle.name}", "Us", logging.getLogger(__name__), gen=9)
    ours, theirs = puzzle.ours, puzzle.theirs
    for line in (
        "|player|p1|Us|",
        "|player|p2|Them|",
        f"|switch|p1a: Ours|{ours.species}, L{ours.level}|{_hp_text(ours.hp)}",
        f"|switch|p2a: Theirs|{theirs.species}, L{theirs.level}|{_hp_text(theirs.hp)}",
    ):
        battle.parse_message(line.split("|"))
    battle.turn = puzzle.turn
    active = battle.active_pokemon
    active.stats = random_battle_stats(active.species, ours.level)
    _apply(active, ours)
    _apply(battle.opponent_active_pokemon, theirs)
    battle._available_moves = [Move(m, gen=9) for m in ours.moves]
    switches = []
    for spec in puzzle.our_reserves:
        mon = Pokemon(gen=9, species=spec.species)
        mon._level = spec.level
        mon.set_hp(_hp_text(spec.hp))
        mon.stats = random_battle_stats(mon.species, spec.level)
        for move in spec.moves:
            mon._moves[move] = Move(move, gen=9)
        _apply(mon, spec)
        battle.team[f"p1: {spec.species}"] = mon
        switches.append(mon)
    battle._available_switches = switches
    battle._team_size = {"p1": 1 + len(switches), "p2": puzzle.their_team_size}
    beliefs = {theirs.species: SetBelief(theirs.species, move_probs=dict.fromkeys(theirs.moves, 1))}
    return battle, beliefs


def _names(mix: dict[str, float]) -> dict[str, float]:
    """'move:earthquake' -> 'earthquake', 'switch:corviknight' -> 'corviknight'."""
    return {label.split(":", 1)[1]: p for label, p in mix.items()}


FLAT = dict.fromkeys(ACTION_CLASSES, 1 / len(ACTION_CLASSES))


def solve(puzzle: Puzzle, forecast: dict | None = None) -> dict:
    """Best response's choice and the minimax mixture's probability on accepted answers.

    forecast defaults to the puzzle's own; pass FLAT or a model's forecast to compare.
    """
    battle, beliefs = build(puzzle)
    forecast = forecast or puzzle.forecast
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PayoffApproximationWarning)
        table = build_payoff_table(battle, beliefs, forecast)
    chosen = best_response(table).chosen.name
    mix = _names(minimax_lp(table, rng=np.random.default_rng(SEED)).mixed_strategy)
    expected = {
        action.name: float(np.dot(row, table.opp_probs))
        for action, row in zip(table.our_actions, table.values, strict=True)
    }
    return {
        "best_response": chosen,
        "best_response_solved": chosen in puzzle.accepted,
        "lp_accepted_mass": sum(p for name, p in mix.items() if name in puzzle.accepted),
        "lp_mix": mix,
        "expected": expected,
    }


GARCHOMP = ("earthquake", "dragonclaw", "stoneedge", "swordsdance")

PUZZLES: tuple[Puzzle, ...] = (
    Puzzle(
        "take_the_ko",
        "A super-effective hit knocks out a weakened Pokemon: take it.",
        Mon("garchomp", GARCHOMP, level=77),
        Mon("heatran", ("magmastorm", "earthpower", "flashcannon"), level=81, hp=0.35),
        frozenset({"earthquake"}),
    ),
    Puzzle(
        "pick_super_effective",
        "Against Steel, use the 2x Ground move, not the resisted Dragon move.",
        Mon("garchomp", GARCHOMP, level=77),
        Mon("gholdengo", ("makeitrain", "shadowball", "nastyplot", "recover"), level=78),
        frozenset({"earthquake"}),
    ),
    Puzzle(
        "ground_cannot_hit_flying",
        "Earthquake does nothing to a Flying type; Fire Fang hits Steel for 2x.",
        Mon("garchomp", ("earthquake", "dragonclaw", "firefang", "swordsdance"), level=77),
        Mon("corviknight", ("bravebird", "bodypress", "roost", "uturn"), level=82),
        frozenset({"firefang", "dragonclaw", "swordsdance"}),  # anything but EQ
    ),
    Puzzle(
        "electric_cannot_hit_ground",
        "Thunderbolt and Volt Switch do nothing to a Ground type; use Shadow Ball.",
        Mon("jolteon", ("thunderbolt", "voltswitch", "shadowball", "calmmind"), level=86),
        Mon("garchomp", ("earthquake", "dragonclaw", "stoneedge"), level=77, hp=0.6),
        frozenset({"shadowball"}),
    ),
    Puzzle(
        "leave_a_4x_weakness",
        "At 70% HP, faster Weavile's 4x Ice moves knock Garchomp out: switch to the Steel type.",
        Mon("garchomp", GARCHOMP, level=77, hp=0.7),
        Mon("weavile", ("tripleaxel", "iceshard", "knockoff", "swordsdance"), level=80),
        frozenset({"corviknight"}),
        our_reserves=(Mon("corviknight", ("bravebird", "bodypress"), level=82),),
        forecast={"attack": 0.8, "non_damaging": 0.1, "switch": 0.1},
    ),
    Puzzle(
        "switch_to_an_immunity",
        "Earthquake from a faster Garchomp knocks out Heatran (4x): switch to the Flying type.",
        Mon("heatran", ("magmastorm", "earthpower", "flashcannon", "stealthrock"), level=81),
        Mon("garchomp", GARCHOMP, level=77),
        frozenset({"corviknight"}),
        our_reserves=(Mon("corviknight", ("bravebird", "bodypress"), level=82),),
        forecast={"attack": 0.85, "non_damaging": 0.1, "switch": 0.05},
    ),
    Puzzle(
        "set_up_when_safe",
        "Blissey can only chip with Seismic Toss and heals off small hits: Swords Dance first.",
        Mon("garchomp", GARCHOMP, level=77),
        Mon("blissey", ("seismictoss", "softboiled"), level=88),
        frozenset({"swordsdance"}),
        forecast=BLISSEY_LIKELY,
    ),
    Puzzle(
        "do_not_set_up_when_dying",
        "At 15% HP Seismic Toss knocks Garchomp out this turn: a boost would be wasted.",
        Mon("garchomp", GARCHOMP, level=77, hp=0.15),
        Mon("blissey", ("seismictoss", "softboiled"), level=88),
        frozenset({"earthquake", "dragonclaw", "stoneedge"}),
        forecast=BLISSEY_LIKELY,
    ),
    Puzzle(
        "hazards_early",
        "Turn 1, five unseen opponents, nothing threatening in front: set Stealth Rock.",
        Mon("garchomp", ("stealthrock", "earthquake", "dragonclaw", "firefang"), level=77),
        Mon("blissey", ("seismictoss", "softboiled"), level=88),
        frozenset({"stealthrock"}),
        forecast=BLISSEY_LIKELY,
    ),
    Puzzle(
        "no_hazards_on_the_last_pokemon",
        "Their last Pokemon will never switch, so Stealth Rock does nothing: attack.",
        Mon("garchomp", ("stealthrock", "earthquake", "dragonclaw", "firefang"), level=77),
        Mon("blissey", ("seismictoss", "softboiled"), level=88),
        frozenset({"earthquake", "dragonclaw", "firefang"}),
        forecast=BLISSEY_LIKELY,
        their_team_size=1,
        turn=30,
    ),
    Puzzle(
        "heal_when_low_and_safe",
        "Moonblast barely scratches Blissey and Seismic Toss is fixed damage: heal with Moonlight.",
        Mon("clefable", ("moonblast", "moonlight", "calmmind", "thunderwave"), level=84, hp=0.35),
        Mon("blissey", ("seismictoss", "softboiled"), level=88),
        frozenset({"moonlight"}),
        forecast=BLISSEY_LIKELY,
    ),
    Puzzle(
        "priority_to_finish",
        "Iron Valiant is faster and knocks us out, but Extreme Speed moves first and finishes it.",
        Mon(
            "dragonite",
            ("extremespeed", "earthquake", "dragondance", "firepunch"),
            level=74,
            hp=0.15,
        ),
        Mon("ironvaliant", ("moonblast", "closecombat", "knockoff"), level=79, hp=0.1),
        frozenset({"extremespeed"}),
    ),
)
