"""Provisional one-turn payoff tables; assumptions are in docs/f4-implementation.md."""

import math
import warnings
from copy import copy
from dataclasses import replace

import numpy as np
from poke_env.battle import Battle, Move
from poke_env.data import GenData, to_id_str

from pokeagent.config import (
    PAYOFF_ATTACK_TEMPERATURE,
    PAYOFF_EFFECT_TURNS,
    PAYOFF_HAZARD_WEIGHT,
    PAYOFF_TERA_COST,
    RANDOM_BATTLE_GENERATION,
    RANDOM_BATTLE_TEAM_SIZE,
    SEED,
)
from pokeagent.decision._payoff_state import (
    PayoffUnavailable,
    entry_damage,
    first_probability,
    hazard_value,
    hidden_boosts,
    lasting_effect,
    move_effect,
    residual,
    scenario_copy,
    substitute_hp,
    unseen_stand_ins,
)
from pokeagent.decision._turn import turn_value
from pokeagent.interfaces import ACTION_CLASSES, Action, BeliefState, PayoffTable, PokemonView
from pokeagent.models.labels import move_class

SWITCH_COLUMN = "__opponent_switch__"  # A modelled column, never an executable action.
_UNSCORED_WEIGHT = 1e-3  # a candidate the exact-move model gave (almost) nothing


class PayoffApproximationWarning(UserWarning):
    """A reminder that the scoring policy is approximate, not a battle simulator."""


def _base_species(species: str) -> str:
    entry = GenData.from_gen(RANDOM_BATTLE_GENERATION).pokedex.get(species, {})
    return to_id_str(entry.get("baseSpecies", "")) or species


def belief_for(species: str, beliefs: BeliefState):
    """The belief about this Pokemon, also after a form change in battle.

    The parser names a Pokemon by the form it first appeared as (palafin), while poke-env
    follows form changes (palafinhero after Zero to Hero; also Mimikyu-Busted, Eiscue-Noice,
    Terapagos, Ogerpon's Tera forms). So fall back to a belief with the same base species.
    """
    if species in beliefs:
        return beliefs[species]
    base = _base_species(species)
    for name, belief in beliefs.items():
        if _base_species(name) == base:
            return belief
    return None


def _opponent_moves(battle: Battle, beliefs: BeliefState) -> list[tuple[Move, float]]:
    opponent = battle.opponent_active_pokemon
    belief = belief_for(opponent.species, beliefs)
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


def _competent_weights(battle, me_key, opp_key, moves):
    """F15: within their attacks, expect the ones that hurt us most.

    A move's weight is still how likely they have it (the belief), now multiplied by
    exp(expected damage to us / PAYOFF_ATTACK_TEMPERATURE). Damage is capped at our current HP,
    so every move that knocks us out counts the same. Without this, a knockout move got only
    its share of the attack probability, and healing into a knockout could look best (failure
    analysis, cause A). Non-damaging moves keep their belief weights.
    """
    scenario = scenario_copy(battle, me_key, opp_key)
    our_hp = scenario.active_pokemon.current_hp_fraction
    weighted = []
    for move, weight in moves:
        try:
            if move_class(move.id) != "attack":
                weighted.append((move, weight))
                continue
            effect = move_effect(scenario, opp_key, me_key, move)
        except (PayoffUnavailable, ValueError):
            weighted.append((move, weight))  # unscorable: keep the belief weight
            continue
        low, high = effect.damage
        hurt = min(our_hp, (low + high) / 2) * effect.hit_chance
        weighted.append((move, weight * math.exp(hurt / PAYOFF_ATTACK_TEMPERATURE)))
    return weighted


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
    battle: Battle,
    beliefs: BeliefState,
    class_probs: dict[str, float] | None,
    scorer=None,
    move_weights: dict[str, float] | None = None,
    unseen_samples: int = 0,
) -> PayoffTable:
    """Compare legal moves/switches against candidate moves and one aggregate switch column.

    Two damage endpoints, misses and ordering ties are averaged AFTER HP capping and KO
    cancellation. Within-class weights use move-inclusion beliefs as a heuristic, not a
    calibrated choice prediction. Impossible classes are removed; missing information raises
    PayoffUnavailable rather than inventing moves or assigning a new uniform forecast.
    None forecasts support minimax. Emits PayoffApproximationWarning for omitted mechanics.
    Our tera (F14): each move also appears as a tera row when tera is still available; the
    opponent's tera is not modelled. Already changed types are retained.
    scorer (V4): a decision.value_utility.ValueScorer; cells are then the change in predicted
    P(win) instead of HP traded plus a knockout bonus.
    move_weights (E8): P(move | they use a move) from the exact-move model (E6). Within each
    action class, the class probability is split by these instead of by belief inclusion
    probabilities. Candidates the model did not score keep a tiny weight.
    unseen_samples (F13): score an attack into a switch to a Pokemon we have not seen against
    this many stand-in species (0 = such an attack does no damage). The agent turns it on
    for minimax only; see PAYOFF_STAND_INS_FOR in config.py.
    """
    if not isinstance(battle, Battle) or battle.gen != 9 or battle.force_switch:
        raise PayoffUnavailable("A table requires a normal Gen 9 singles turn")
    me = battle.active_pokemon
    if me is None or me.fainted:
        raise PayoffUnavailable("Both active Pokemon must be alive and revealed")
    me_key = next(k for k, mon in battle.team.items() if mon is me)
    return _table(
        battle,
        beliefs,
        class_probs,
        me_key,
        _rows(battle, me_key),
        entering=False,
        scorer=scorer,
        move_weights=move_weights,
        unseen_samples=unseen_samples,
    )


def our_tera_type(battle: Battle) -> str | None:
    """The type our active Pokemon would become if it terastallized now, or None.

    Tera is allowed once per battle. Our own tera type is in the server's request for our team
    (our private knowledge, used for our own options only, never as a model input). Stellar
    (Terapagos only) is left out: it does not change the Pokemon's defensive type.
    """
    me = battle.active_pokemon
    if not battle.can_tera or me is None or me.is_terastallized:
        return None
    request = getattr(me, "_last_request", None) or {}
    name = request.get("teraType") or (me.tera_type.name if me.tera_type else None)
    if not name or name.lower() == "stellar":
        return None
    return name.lower()


def _rows(battle: Battle, me_key: str) -> list:
    """Our legal moves, the same moves with tera (F14) if allowed, then switches."""
    rows = [(Action("move", m.id), me_key, m) for m in battle.available_moves]
    if our_tera_type(battle):
        rows += [(Action("move", m.id, tera=True), me_key, m) for m in battle.available_moves]
    for mon in battle.available_switches:
        if mon.max_hp <= 0 or mon.current_hp_fraction <= 0:
            continue
        key = next(k for k, candidate in battle.team.items() if candidate is mon)
        rows.append((Action("switch", mon.species), key, None))
    return rows


def _view(mon) -> PokemonView:
    """The public facts the value model reads about a Pokemon that is about to come in."""
    return PokemonView(
        species=mon.species,
        level=mon.level,
        hp_fraction=mon.current_hp_fraction,
        status=mon.status.name.lower() if mon.status else None,
    )


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


def _table(
    battle,
    beliefs,
    class_probs,
    me_key,
    rows,
    *,
    entering,
    lookahead=True,
    scorer=None,
    move_weights=None,
    unseen_samples=0,
):
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
    if not rows or len({(a.kind, a.name, a.tera) for a, _, _ in rows}) != len(rows):
        raise PayoffUnavailable("Legal actions are empty or have duplicate labels")
    moves = _opponent_moves(battle, beliefs)
    if move_weights:
        # The candidate list stays the same; only how the forecast is shared among them changes.
        moves = [(m, max(move_weights.get(m.id, 0.0), _UNSCORED_WEIGHT)) for m, _ in moves]
    elif PAYOFF_ATTACK_TEMPERATURE:
        moves = _competent_weights(battle, me_key, opp_key, moves)
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
    # F13: an unseen switch-in is scored as an average over stand-in species from the Random
    # Battle pool, so an attack into a switch gets credit for the damage it would do. The
    # stand-ins live only on a private copy of the battle used to score cells.
    stage, stand_ins = battle, {}
    if unseen and unseen_samples:
        stand_ins = unseen_stand_ins(battle, unseen_samples, SEED)
    if stand_ins:
        stage = copy(battle)
        stage._opponent_team = {**battle.opponent_team, **stand_ins}
        reserves.extend((key, unseen / len(stand_ins)) for key in stand_ins)
    elif unseen:
        reserves.append((None, unseen))
    belief = belief_for(opp.species, beliefs)
    hidden = hidden_boosts(opp, belief)
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

    def next_turn_value(state: Battle, rows=None) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PayoffApproximationWarning)
            inner = _table(
                state,
                beliefs,
                class_probs,
                me_key,
                next_rows if rows is None else rows,
                entering=False,
                lookahead=False,
                scorer=scorer,
                move_weights=move_weights,
                unseen_samples=unseen_samples,
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
                    hazard = PAYOFF_HAZARD_WEIGHT * hazard_value(battle, effect.battle, unseen)
                    # In value mode, convert HP to P(win) at the current position's rate.
                    value += hazard * (scorer.hp_slope() if scorer else 1.0)
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

    tera_type = our_tera_type(battle) if not entering else None
    tera_terms: list[float] = []

    def tera_term() -> float:
        """F14: what terastallizing adds beyond this turn, the same for every tera row.

        Tera lasts for the rest of the game, so (as for other lasting effects) it is worth the
        change in next turn's table: our new type against their moves, minus the tera options
        we give up. That one-turn look cannot see that tera might be better saved for another
        Pokemon later, so a fixed cost PAYOFF_TERA_COST (in HP, converted like hazards) is
        subtracted as well.
        """
        if not tera_terms:
            # The cost is in HP; in value mode convert it to P(win) as for hazards.
            value = -PAYOFF_TERA_COST * (scorer.hp_slope() if scorer else 1.0)
            if lookahead and PAYOFF_EFFECT_TURNS:
                state = scenario_copy(battle, me_key, opp_key)
                state.active_pokemon.terastallize(tera_type)
                state._can_tera = False
                try:
                    if not base:
                        base.append(next_turn_value(battle))
                    plain = [row for row in next_rows if not row[0].tera]
                    value += PAYOFF_EFFECT_TURNS * (next_turn_value(state, plain) - base[0])
                except PayoffUnavailable:
                    pass
            tera_terms.append(value)
        return tera_terms[0]

    def with_future(effect, lasting, target_is_the_active_one):
        if lasting is None:
            return effect
        value, needs_user, needs_target = lasting
        if needs_target and not target_is_the_active_one:
            return effect  # aimed at a Pokemon that is not the one we evaluated
        return replace(
            effect, future=value, future_needs_user=needs_user, future_needs_target=needs_target
        )

    def cell(our_key, their_key, our_move, their_move, tera=False):
        scenario = scenario_copy(stage, our_key, their_key)
        their_unseen = their_key is None or their_key in stand_ins
        our_mon, their_mon = scenario.active_pokemon, scenario.opponent_active_pokemon
        if tera:
            # Tera happens before anyone moves: it changes our STAB and our weaknesses.
            our_mon.terastallize(tera_type)
        losses = (
            entry_damage(our_mon, scenario, scenario.side_conditions)
            if entering or our_key != me_key
            else 0,
            entry_damage(their_mon, scenario, scenario.opponent_side_conditions)
            if their_key is not None and their_key != opp_key
            else 0,
        )
        ours = move_effect(scenario, our_key, their_key, our_move)
        # #119: only the opponent's active Pokemon has a belief to weigh hidden boosts by.
        boosts = hidden if their_key == opp_key else None
        theirs = (
            move_effect(scenario, their_key, our_key, their_move, boosts)
            if their_key
            else move_effect(scenario, our_key, None, None)
        )
        if our_key == me_key:
            ours = with_future(ours, future(me_key, opp_key, our_move), their_key == opp_key)
        if their_key == opp_key:
            theirs = with_future(theirs, future(opp_key, me_key, their_move), our_key == me_key)
        value_terms = {}
        if scorer is not None:
            value_terms = {
                # For an unseen switch-in the value model keeps the current matchup with their
                # HP at full (it cannot read a Pokemon that has not been revealed); the stand-in
                # only decides how much damage our move does.
                "end_value": scorer.end_value(
                    None if our_key == me_key else _view(our_mon),
                    None if their_key == opp_key or their_unseen else _view(their_mon),
                    unseen=their_unseen,
                ),
                "start_value": scorer.win_probability(),
            }
        lasting_tera = tera_term() if tera else 0.0
        return lasting_tera + turn_value(
            our_mon.current_hp_fraction,
            their_mon.current_hp_fraction if their_mon else 1,
            ours,
            theirs,
            first_probability(scenario, our_move, their_move, boosts),
            losses,
            residuals=(residual(our_mon, scenario), residual(their_mon, scenario)),
            # A Substitute leaves with its owner, so only the Pokemon staying in keeps one.
            substitutes=(
                substitute_hp(our_mon) if our_key == me_key else 0.0,
                substitute_hp(their_mon) if their_key == opp_key else 0.0,
            ),
            **value_terms,
        )

    values = []
    try:
        for action, our_key, our_move in rows:
            tera = action.tera
            row = [cell(our_key, opp_key, our_move, move, tera) for move, _ in moves]
            if reserves:
                weight = sum(w for _, w in reserves)
                row.append(
                    sum(w * cell(our_key, target, our_move, None, tera) for target, w in reserves)
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
        "secondary effects, switch-in abilities, the opponent's tera, Grassy Terrain, Leech Seed "
        "or the HP left in a damaged Substitute. "
        "Candidate moves need not be exhaustive. Unseen switch targets receive zero outgoing "
        "credit unless stand-in species are requested. See docs/f4-implementation.md.",
        PayoffApproximationWarning,
        stacklevel=2,
    )
    return PayoffTable([a for a, _, _ in rows], columns, values, probs)
