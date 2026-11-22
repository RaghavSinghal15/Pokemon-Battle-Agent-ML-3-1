"""Score a turn's outcome by the change in predicted P(win) (V4, #74).

The hand-made utility counts HP fractions plus a knockout bonus. Here an outcome is worth
P(win at the end of the turn) - P(win now), from the logistic-regression value model (V2).

Why logistic regression and not LightGBM: a payoff table has dozens of cells, each with up to
32 outcome branches, plus lookahead tables. Logistic regression turns features into
logit = b + sum(w_i * x_i), so a change in HP moves the logit by w * (change): one
multiplication per branch. LightGBM would need a full model call per branch. Its validation
log loss is close (0.590 vs 0.581).

Only the HP-related features change within a turn: the active Pokemon's HP, team HP and
Pokemon remaining. Everything else (types, speed, beliefs, boosts) is read from the position
at the start of the turn, for the pair of Pokemon that will be facing each other.
"""

import math
from dataclasses import replace
from functools import lru_cache

from poke_env.data import GenData

from pokeagent.config import RANDOM_BATTLE_GENERATION
from pokeagent.features.value import VALUE_FEATURE_NAMES, build_value_features
from pokeagent.interfaces import PokemonView, PublicState, SideView

_CERTAIN = 30.0  # a logit this large means P(win) is 0 or 1 to within 1e-13


@lru_cache(maxsize=1)
def _dex() -> GenData:
    return GenData.from_gen(RANDOM_BATTLE_GENERATION)


def _sigmoid(logit: float) -> float:
    return 1 / (1 + math.exp(-max(-_CERTAIN, min(_CERTAIN, logit))))


class ValueScorer:
    """P(win) for end-of-turn HP, from our side of a start-of-turn public state."""

    def __init__(self, model, state: PublicState):
        """model: a models.value.ValueModel wrapping StandardScaler + LogisticRegression."""
        pipeline = model.estimator
        scaler, regression = pipeline[0], pipeline[-1]
        if list(model.columns) != list(VALUE_FEATURE_NAMES):
            raise ValueError("The value utility needs the logistic model on all value features")
        # Undo the standardisation, so the weights apply to raw feature values.
        coefficients = regression.coef_[0] / scaler.scale_
        self.weights = dict(zip(model.columns, coefficients, strict=True))
        self.intercept = float(regression.intercept_[0] - (coefficients * scaler.mean_).sum())
        self.state = state
        self._rows: dict[tuple, dict] = {}

    def logit(self, row: dict) -> float:
        return self.intercept + sum(self.weights[name] * row[name] for name in self.weights)

    def win_probability(self) -> float:
        return _sigmoid(self.logit(self._row(None, None)))

    def hp_slope(self) -> float:
        """Change in P(win) per Pokemon's worth of team HP, at the current position: converts
        HP-valued terms (hazard damage still to come) into the same units."""
        p = self.win_probability()
        per_team_hp = (abs(self.weights["my_team_hp"]) + abs(self.weights["opp_team_hp"])) / 2
        return p * (1 - p) * per_team_hp

    def _row(self, ours: PokemonView | None, theirs: PokemonView | None) -> dict:
        """Value features with these two Pokemon active (None = as now). Cached."""
        key = (ours and ours.species, ours and ours.hp_fraction, theirs and theirs.species)
        if key not in self._rows:
            state = self.state
            if ours is not None:
                state = replace(state, me=_with_active(state.me, ours))
            if theirs is not None:
                state = replace(state, opp=_with_active(state.opp, theirs))
            self._rows[key] = build_value_features(state, _dex())
        return self._rows[key]

    def end_value(self, ours: PokemonView | None, theirs: PokemonView | None, unseen=False):
        """A function (our HP, their HP at the end of the turn) -> P(win).

        ours / theirs are the Pokemon that will be facing each other this turn (None = the
        active one now). unseen=True means they switch to a Pokemon we have not seen: its
        details are unknown, so the current matchup is kept with their HP set to full.
        """
        row = dict(self._row(ours, theirs))
        if unseen:
            row["opp_hp"] = 1.0
        base = self.logit(row)

        def value(our_hp: float, their_hp: float) -> float:
            return _sigmoid(
                base + self._change(row, "my", our_hp) + self._change(row, "opp", their_hp)
            )

        return value

    def _change(self, row: dict, side: str, end_hp: float) -> float:
        """Logit change when this side's active Pokemon ends the turn on end_hp."""
        w = self.weights
        start = row[f"{side}_hp"]
        if end_hp > 0:
            return (w[f"{side}_hp"] + w[f"{side}_team_hp"]) * (end_hp - start)
        left = row[f"{side}_remaining"] - 1
        if left <= 0:  # their last Pokemon fainted: the game is decided
            return -2 * _CERTAIN if side == "my" else 2 * _CERTAIN
        team_after = row[f"{side}_team_hp"] - start
        # The replacement is not known yet: give it the average HP of the rest of the team.
        replacement_hp = team_after / left
        return (
            w[f"{side}_hp"] * (replacement_hp - start)
            + w[f"{side}_team_hp"] * (team_after - row[f"{side}_team_hp"])
            - w[f"{side}_remaining"]
        )


def _with_active(side: SideView, mon: PokemonView) -> SideView:
    """The same side with `mon` as the active Pokemon (added if it was not revealed yet)."""
    pokemon = [replace(p, active=p.species == mon.species) for p in side.pokemon]
    if not any(p.species == mon.species for p in pokemon):
        pokemon.append(replace(mon, active=True))
    return replace(side, pokemon=pokemon)
