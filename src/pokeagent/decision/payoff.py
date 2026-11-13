"""Provisional one-turn payoff tables; assumptions are in docs/f4-implementation.md."""

import math
import warnings

from poke_env.battle import Battle, Move, MoveCategory

from pokeagent.config import RANDOM_BATTLE_TEAM_SIZE
from pokeagent.decision._payoff_state import (
    PayoffUnavailable,
    entry_damage,
    first_probability,
    move_effect,
    scenario_copy,
)
from pokeagent.decision._turn import turn_value
from pokeagent.interfaces import ACTION_CLASSES, Action, BeliefState, PayoffTable

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


def _probabilities(moves, switch_available, class_probs):
    if class_probs is None:
        return None
    if set(class_probs) != set(ACTION_CLASSES):
        raise PayoffUnavailable("Forecast must contain exactly the three action classes")
    if any(not math.isfinite(p) or p < 0 for p in class_probs.values()):
        raise PayoffUnavailable("Class probabilities must be finite and non-negative")
    total = sum(class_probs.values())
    if not math.isclose(total, 1, rel_tol=0, abs_tol=1e-6):
        raise PayoffUnavailable("Class probabilities must sum to one")
    classes = ["non_damaging" if m.category == MoveCategory.STATUS else "attack" for m, _ in moves]
    weights = {
        c: sum(w for (_, w), name in zip(moves, classes, strict=True) if name == c)
        for c in ACTION_CLASSES[:2]
    }
    for name, weight in weights.items():
        if class_probs[name] > 0 and weight == 0:
            raise PayoffUnavailable(f"Positive {name} probability has no move candidates")
    if class_probs["switch"] > 0 and not switch_available:
        raise PayoffUnavailable("Positive switch probability has no possible reserve")
    result = [
        class_probs[c] / total * w / weights[c] for (_, w), c in zip(moves, classes, strict=True)
    ]
    if switch_available:
        result.append(class_probs["switch"] / total)
    return result


def build_payoff_table(
    battle: Battle, beliefs: BeliefState, class_probs: dict[str, float] | None
) -> PayoffTable:
    """Compare legal moves/switches against candidate moves and one aggregate switch column.

    Two damage endpoints, misses and ordering ties are averaged AFTER HP capping and KO
    cancellation. Within-class weights use move-inclusion beliefs as a heuristic, not a
    calibrated choice prediction. Missing positive-mass candidates raise PayoffUnavailable;
    None forecasts support minimax. Emits PayoffApproximationWarning for omitted mechanics.
    New voluntary tera options are deferred; already changed types are retained.
    """
    if not isinstance(battle, Battle) or battle.gen != 9 or battle.force_switch:
        raise PayoffUnavailable("A table requires a normal Gen 9 singles turn")
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    if me is None or opp is None or me.fainted or opp.fainted:
        raise PayoffUnavailable("Both active Pokemon must be alive and revealed")
    me_key = next(k for k, mon in battle.team.items() if mon is me)
    opp_key = next(k for k, mon in battle.opponent_team.items() if mon is opp)
    rows = [(Action("move", m.id), me_key, m) for m in battle.available_moves]
    for mon in battle.available_switches:
        key = next(k for k, candidate in battle.team.items() if candidate is mon)
        rows.append((Action("switch", mon.species), key, None))
    if not rows or len({(a.kind, a.name) for a, _, _ in rows}) != len(rows):
        raise PayoffUnavailable("Legal actions are empty or have duplicate labels")
    moves = _opponent_moves(battle, beliefs)
    if not moves:
        raise PayoffUnavailable(
            "No opponent move candidates; supply public priors or revealed moves"
        )
    reserves = [
        (k, 1) for k, mon in battle.opponent_team.items() if mon is not opp and not mon.fainted
    ]
    # Observed |teamsize| metadata, never the opponent's hidden team request.
    team_size = battle._team_size.get(battle.opponent_role, RANDOM_BATTLE_TEAM_SIZE)
    unseen = max(0, team_size - len(battle.opponent_team))
    if unseen:
        reserves.append((None, unseen))
    probs = _probabilities(moves, bool(reserves), class_probs)
    columns = [Action("move", m.id) for m, _ in moves]
    if reserves:
        columns.append(Action("switch", SWITCH_COLUMN))
    if not columns:
        raise PayoffUnavailable("No plausible opponent actions are available")

    def cell(our_key, their_key, our_move, their_move):
        scenario = scenario_copy(battle, our_key, their_key)
        our_mon, their_mon = scenario.active_pokemon, scenario.opponent_active_pokemon
        losses = (
            entry_damage(our_mon, scenario, scenario.side_conditions) if our_key != me_key else 0,
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
        return turn_value(
            our_mon.current_hp_fraction,
            their_mon.current_hp_fraction if their_mon else 1,
            ours,
            theirs,
            first_probability(scenario, our_move, their_move),
            losses,
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
        "no secondary/residual/setup utility, switch-in abilities or new tera choices. "
        "Candidate moves need not be exhaustive. Unseen switch targets receive zero outgoing "
        "credit. See docs/f4-implementation.md.",
        PayoffApproximationWarning,
        stacklevel=2,
    )
    return PayoffTable([a for a, _, _ in rows], columns, values, probs)
