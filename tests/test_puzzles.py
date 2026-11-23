"""Puzzle positions (V3, #73) and the healing amounts they exposed."""

import pytest
from poke_env.battle import Move, Status, Weather

from pokeagent.decision._payoff_state import heal_fraction, lasting_effect, scenario_copy
from pokeagent.eval.puzzles import PUZZLES, Puzzle, build, solve

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.mark.parametrize("puzzle", PUZZLES, ids=lambda p: p.name)
def test_every_puzzle_builds_and_its_answers_are_legal(puzzle: Puzzle):
    battle, beliefs = build(puzzle)
    legal = {m.id for m in battle.available_moves} | {p.species for p in battle.available_switches}
    assert puzzle.accepted <= legal
    assert set(beliefs[puzzle.theirs.species].move_probs) == set(puzzle.theirs.moves)
    assert sum(puzzle.forecast.values()) == pytest.approx(1)
    assert 0 <= solve(puzzle)["lp_accepted_mass"] <= 1 + 1e-9


def test_the_decision_engine_still_solves_the_puzzles_it_solved():
    """A floor, not a target: a change that loses a puzzle should say why in its PR.

    These run with the HP utility (CI has no value model). F13 lost leave_a_4x_weakness here:
    Stone Edge now gets credit against a possible switch-in, and it beats switching out by
    0.018. With the value utility the agent uses, all 12 are still solved.
    """
    solved = {p.name for p in PUZZLES if solve(p)["best_response_solved"]}
    assert len(solved) >= 10, f"unsolved: {sorted(p.name for p in PUZZLES if p.name not in solved)}"


def clefable(weather=None):
    puzzle = next(p for p in PUZZLES if p.name == "heal_when_low_and_safe")
    battle, _ = build(puzzle)
    if weather:
        battle._weather = {weather: 1}
    return scenario_copy(battle, "p1: Ours", "p2: Theirs")


def test_weather_dependent_healing():
    def heal(move, weather=None):
        return heal_fraction(clefable(weather), "p1: Ours", "p2: Theirs", Move(move, gen=9))

    assert heal("moonlight") == pytest.approx(1 / 2)
    assert heal("moonlight", Weather.SUNNYDAY) == pytest.approx(2 / 3)
    assert heal("synthesis", Weather.RAINDANCE) == pytest.approx(1 / 4)
    assert heal("shoreup", Weather.SANDSTORM) == pytest.approx(2 / 3)
    assert heal("softboiled") == pytest.approx(1 / 2)  # stated in the move data already
    assert heal("rest") == 1.0
    battle = clefable()
    blissey = battle.get_pokemon("p2: Theirs")
    clef = battle.get_pokemon("p1: Ours")
    expected = blissey.stats["atk"] / clef.stats["hp"]
    assert heal_fraction(battle, "p1: Ours", "p2: Theirs", Move("strengthsap", gen=9)) == (
        pytest.approx(expected)
    )


def test_rest_puts_the_user_to_sleep_unless_at_full_hp():
    battle = clefable()
    after = lasting_effect(battle, "p1: Ours", "p2: Theirs", Move("rest", gen=9))
    assert after.battle.get_pokemon("p1: Ours").status == Status.SLP and after.needs_user
    battle.get_pokemon("p1: Ours").set_hp("100/100")
    assert lasting_effect(battle, "p1: Ours", "p2: Theirs", Move("rest", gen=9)) is None
