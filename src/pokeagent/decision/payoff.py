"""Provisional one-turn payoff tables; assumptions are in docs/f4-implementation.md."""

import math
import warnings
from copy import copy
from dataclasses import replace

import numpy as np
from poke_env.battle import Battle, Move

from pokeagent.config import PAYOFF_EFFECT_TURNS, PAYOFF_HAZARD_WEIGHT, RANDOM_BATTLE_TEAM_SIZE
from pokeagent.decision._payoff_state import (
    PayoffUnavailable,
    entry_damage,
    first_probability,
    hazard_value,
    lasting_effect,
    move_effect,
    residual,
    scenario_copy,
    substitute_hp,
)
from pokeagent.decision._turn import turn_value
from pokeagent.interfaces import ACTION_CLASSES, Action, BeliefState, PayoffTable
from pokeagent.models.labels import move_class

SWITCH_COLUMN = "__opponent_switch__"  # A modelled column, never an executable action.


class PayoffApproximationWarning(UserWarning):
    """A reminder that the scoring policy is approximate, not a battle simulator."""


def _opponent_moves(battle: Battle, beliefs: BeliefState) -> list[tuple[Move, float]]:
    opponent = battle.opponent_active_pokemon
    belief = beliefs.get(opponent.species)
    weights = dict(belief.move_probs) if belief else {}
    for weight in weights.values():
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise PayoffUnavailable("Move inclusion probabilities must lie between zero and one")
    for name in opponent.moves:
        weights[name] = 1.0  # Revelations override contradictory priors; no peer data is read.
    try:
        return [
            (Move(name, gen=battle.gen), weight)
            for name, weight in sorted(weights.items())
            if weight > 0
        ]
    except Exception as error:
        raise PayoffUnavailable(f"Invalid opponent move candidate: {error}") from error


def _probabilities(moves, switch_available, class_probs, complete_move_pool):
    if class_probs is None:
        return None
    if set(class_probs) != set(ACTION_CLASSES):
        raise PayoffUnavailable("Forecast must contain exactly the three action classes")
    if any(not math.isfinite(p) or p < 0 for p in class_probs.values()):
        raise PayoffUnavailable("Class probabilities must be finite and non-negative")
    total = sum(class_probs.values())
    if not math.isclose(total, 1, rel_tol=0, abs_tol=1e-6):
        raise PayoffUnavailable("Class probabilities must sum to one")
    # Same grouping the opponent model was trained on (one shared rule in models/labels.py).
    try:
        classes = [move_class(m.id) for m, _ in moves]
    except ValueError as error:
        raise PayoffUnavailable(str(error)) from error
    weights = {
        c: sum(w for (_, w), name in zip(moves, classes, strict=True) if name == c)
        for c in ACTION_CLASSES[:2]
    }
    mass = dict(class_probs)
    for name, weight in weights.items():
        if mass[name] > 0 and weight == 0:
            if not complete_move_pool:
                raise PayoffUnavailable(f"Positive {name} probability has no move candidates")
            mass[name] = 0.0
    if not switch_available:
        mass["switch"] = 0.0
    total = sum(mass.values())
    if total == 0:
        raise PayoffUnavailable("Forecast assigns no mass to available actions")
    result = [mass[c] / total * w / weights[c] for (_, w), c in zip(moves, classes, strict=True)]
    if switch_available:
        result.append(mass["switch"] / total)
    return result


def build_payoff_table(
    battle: Battle, beliefs: BeliefState, class_probs: dict[str, float] | None
) -> PayoffTable:
    """Compare legal moves/switches against candidate moves and one aggregate switch column.

    Two damage endpoints, misses and ordering ties are averaged AFTER HP capping and KO
    cancellation. Within-class weights use move-inclusion beliefs as a heuristic, not a
    calibrated choice prediction. Impossible classes are removed; missing information raises
    PayoffUnavailable rather than inventing moves or assigning a new uniform forecast.
    None forecasts support minimax. Emits PayoffApproximationWarning for omitted mechanics.
    New voluntary tera options are deferred; already changed types are retained.
    """
    if not isinstance(battle, Battle) or battle.gen != 9 or battle.force_switch:
        raise PayoffUnavailable("A table requires a normal Gen 9 singles turn")
    me = battle.active_pokemon
    if me is None or me.fainted:
        raise PayoffUnavailable("Both active Pokemon must be alive and revealed")
    me_key = next(k for k, mon in battle.team.items() if mon is me)
    return _table(battle, beliefs, class_probs, me_key, _rows(battle, me_key), entering=False)


def _rows(battle: Battle, me_key: str) -> list:
    """Our legal moves, then switches to living reserves."""
    rows = [(Action("move", m.id), me_key, m) for m in battle.available_moves]
    for mon in battle.available_switches:
        if mon.max_hp <= 0 or mon.current_hp_fraction <= 0:
            continue
        key = next(k for k, candidate in battle.team.items() if candidate is mon)
        rows.append((Action("switch", mon.species), key, None))
    return rows


def _table_value(table: PayoffTable) -> float:
    """What a table is worth to us: best expected row, or the best guaranteed row if there
    is no forecast (minimax mode)."""
    values = np.asarray(table.values, dtype=float)
    if table.opp_probs is None:
        return float(values.min(axis=1).max())
    return float((values @ np.asarray(table.opp_probs, dtype=float)).max())


def replacement_tables(
    battle: Battle, beliefs: BeliefState, class_probs: dict[str, float] | None
) -> dict[str, PayoffTable]:
    """For a forced replacement, next turn's table as if each legal reserve came in.

    Rows are the reserve's own known moves (our team's moves are not hidden from us), and
    every row pays the reserve's hazard damage on entry. Columns, beliefs and scoring are
    exactly those of build_payoff_table, so type matchups, abilities, items and Speed enter
    through the same damage and order calculations. Our switch rows are left out: the
    question is how good each Pokemon is to have in battle. Keys are species. Raises
    PayoffUnavailable when no candidate can be scored (for example, when the opponent's
    active Pokemon also fainted and its replacement is not yet known).
    """
    if not isinstance(battle, Battle) or battle.gen != 9:
        raise PayoffUnavailable("Replacement scoring requires a Gen 9 singles battle")
    tables, reasons = {}, []
    for mon in battle.available_switches:
        if mon.max_hp <= 0 or mon.current_hp_fraction <= 0:
            continue
        key = next(k for k, candidate in battle.team.items() if candidate is mon)
        moves = [Move(move_id, gen=battle.gen) for move_id in mon.moves]
        rows = [(Action("move", move.id), key, move) for move in moves]
        try:
            tables[mon.species] = _table(battle, beliefs, class_probs, key, rows, entering=True)
        except PayoffUnavailable as error:
            reasons.append(f"{mon.species}: {error}")
    if not tables:
        raise PayoffUnavailable("No replacement could be scored; " + "; ".join(reasons))
    return tables


def _table(battle, beliefs, class_probs, me_key, rows, *, entering, lookahead=True):
    """Score rows for our Pokemon at me_key against the opponent's active Pokemon.

    entering=True charges hazard damage on every row, because that Pokemon is arriving.
    lookahead=True values lasting effects (stat changes, status, hazards) by how much they
    change next turn's table (F8); the inner tables use lookahead=False, so it goes one turn
    deep only.
    """
    opp = battle.opponent_active_pokemon
    if opp is None or opp.fainted:
        raise PayoffUnavailable("Both active Pokemon must be alive and revealed")
    opp_key = next(k for k, mon in battle.opponent_team.items() if mon is opp)
    if not rows or len({(a.kind, a.name) for a, _, _ in rows}) != len(rows):
        raise PayoffUnavailable("Legal actions are empty or have duplicate labels")
    moves = _opponent_moves(battle, beliefs)
    if not moves:
        raise PayoffUnavailable(
            "No opponent move candidates; supply public priors or revealed moves"
        )
    # Illusion can leave a 0/0-HP disguise object behind. It is not a real team member.
    observed = {k: mon for k, mon in battle.opponent_team.items() if mon.max_hp > 0}
    reserves = [
        (k, 1)
        for k, mon in observed.items()
        if mon is not opp and not mon.fainted and mon.current_hp_fraction > 0
    ]
    # Observed |teamsize| metadata, never the opponent's hidden team request.
    team_size = battle._team_size.get(battle.opponent_role, RANDOM_BATTLE_TEAM_SIZE)
    unseen = max(0, team_size - len(observed))
    if unseen:
        reserves.append((None, unseen))
    belief = beliefs.get(opp.species)
    probs = _probabilities(moves, bool(reserves), class_probs, bool(belief and belief.move_probs))
    columns = [Action("move", m.id) for m, _ in moves]
    if reserves:
        columns.append(Action("switch", SWITCH_COLUMN))
    if not columns:
        raise PayoffUnavailable("No plausible opponent actions are available")

    # Lasting effects: value = (next turn's table value with the effect - without it), times
    # PAYOFF_EFFECT_TURNS. Next turn is approximated from the current position (HP as now).
    # With both weights at zero no lasting effect can count, so skip the extra work.
    lookahead = lookahead and bool(PAYOFF_EFFECT_TURNS or PAYOFF_HAZARD_WEIGHT)
    futures: dict[tuple[str, str], tuple[float, bool, bool] | None] = {}
    base: list[float] = []
    # Next turn's rows come from the real battle: the copies made for lasting effects have
    # copied Pokemon objects, but the same keys, moves and legal switches.
    next_rows = _rows(battle, me_key) if lookahead and not entering else []

    def next_turn_value(state: Battle) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PayoffApproximationWarning)
            inner = _table(
                state, beliefs, class_probs, me_key, next_rows, entering=False, lookahead=False
            )
        return _table_value(inner)

    def future(user_key, target_key, move):
        if not lookahead or entering or move is None:
            return None
        key = (user_key, move.id)
        if key not in futures:
            effect = lasting_effect(battle, user_key, target_key, move)
            futures[key] = None
            if effect is not None:
                value = 0.0
                if effect.changes_field:
                    value += PAYOFF_HAZARD_WEIGHT * hazard_value(battle, effect.battle, unseen)
                if effect.changes_pokemon and PAYOFF_EFFECT_TURNS:
                    # Look one turn ahead with the stat/status change but the hazards as now
                    # (hazards are valued above, over all remaining switch-ins).
                    state = copy(effect.battle)
                    state._side_conditions = dict(battle.side_conditions)
                    state._opponent_side_conditions = dict(battle.opponent_side_conditions)
                    try:
                        if not base:
                            base.append(next_turn_value(battle))
                        value += PAYOFF_EFFECT_TURNS * (next_turn_value(state) - base[0])
                    except PayoffUnavailable:
                        pass
                futures[key] = (value, effect.needs_user, effect.needs_target)
        return futures[key]

    def with_future(effect, lasting, target_is_the_active_one):
        if lasting is None:
            return effect
        value, needs_user, needs_target = lasting
        if needs_target and not target_is_the_active_one:
            return effect  # aimed at a Pokemon that is not the one we evaluated
        return replace(
            effect, future=value, future_needs_user=needs_user, future_needs_target=needs_target
        )

    def cell(our_key, their_key, our_move, their_move):
        scenario = scenario_copy(battle, our_key, their_key)
        our_mon, their_mon = scenario.active_pokemon, scenario.opponent_active_pokemon
        losses = (
            entry_damage(our_mon, scenario, scenario.side_conditions)
            if entering or our_key != me_key
            else 0,
            entry_damage(their_mon, scenario, scenario.opponent_side_conditions)
            if their_key is not None and their_key != opp_key
            else 0,
        )
        ours = move_effect(scenario, our_key, their_key, our_move)
        theirs = (
            move_effect(scenario, their_key, our_key, their_move)
            if their_key
            else move_effect(scenario, our_key, None, None)
        )
        if our_key == me_key:
            ours = with_future(ours, future(me_key, opp_key, our_move), their_key == opp_key)
        if their_key == opp_key:
            theirs = with_future(theirs, future(opp_key, me_key, their_move), our_key == me_key)
        return turn_value(
            our_mon.current_hp_fraction,
            their_mon.current_hp_fraction if their_mon else 1,
            ours,
            theirs,
            first_probability(scenario, our_move, their_move),
            losses,
            residuals=(residual(our_mon, scenario), residual(their_mon, scenario)),
            # A Substitute leaves with its owner, so only the Pokemon staying in keeps one.
            substitutes=(
                substitute_hp(our_mon) if our_key == me_key else 0.0,
                substitute_hp(their_mon) if their_key == opp_key else 0.0,
            ),
        )

    values = []
    try:
        for _, our_key, our_move in rows:
            row = [cell(our_key, opp_key, our_move, move) for move, _ in moves]
            if reserves:
                weight = sum(w for _, w in reserves)
                row.append(
                    sum(w * cell(our_key, target, our_move, None) for target, w in reserves)
                    / weight
                )
            values.append(row)
    except PayoffUnavailable:
        raise
    except Exception as error:
        raise PayoffUnavailable(f"Could not score this turn: {error}") from error
    warnings.warn(
        "Approximate payoffs: two damage endpoints and limited accuracy/order modifiers; "
        "lasting effects valued one turn ahead from the current HP; no chance-based "
        "secondary effects, switch-in abilities, new tera choices, Grassy Terrain, Leech Seed "
        "or the HP left in a damaged Substitute. "
        "Candidate moves need not be exhaustive. Unseen switch targets receive zero outgoing "
        "credit. See docs/f4-implementation.md.",
        PayoffApproximationWarning,
        stacklevel=2,
    )
    return PayoffTable([a for a, _, _ in rows], columns, values, probs)
