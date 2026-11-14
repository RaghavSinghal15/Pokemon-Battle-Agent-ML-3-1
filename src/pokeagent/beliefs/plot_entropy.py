"""Plot uncertainty about an opponent's role using a local spectator replay."""

import argparse
import json
from math import log2
from pathlib import Path

import matplotlib
from poke_env.battle import Battle

from pokeagent.beliefs.tracker import prior_belief, update_belief
from pokeagent.config import INTERIM_DIR, REPORTS_DIR
from pokeagent.interfaces import PokemonView

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def role_entropy(role_probs: dict[str, float]) -> float:
    """Measure role uncertainty in bits, with zero for a single certain role."""
    return sum(
        -probability * log2(probability) for probability in role_probs.values() if probability > 0
    )


def _role_traces(record: dict, username: str) -> dict[str, list[tuple[int, float, list[str]]]]:
    # Import the repo's public replay spike here so unit tests need no replay fixture.
    from scripts.spikes.replay_parse import parse_log

    traces: dict[str, list[tuple[int, float, list[str]]]] = {}

    def on_turn(battle: Battle) -> None:
        pokemon = battle.opponent_active_pokemon
        if pokemon is None:
            return
        prior = prior_belief(pokemon.species)
        if len(prior.role_probs) < 2:
            return
        seen = PokemonView(
            species=pokemon.species,
            level=pokemon.level,
            revealed_moves=list(pokemon.moves),
            revealed_ability=pokemon.ability,
            tera_type=pokemon.tera_type.name if pokemon.tera_type else None,
            terastallized=pokemon.is_terastallized,
        )
        belief = update_belief(prior, seen)
        trace = traces.setdefault(pokemon.species, [])
        trace.append((battle.turn, role_entropy(belief.role_probs), sorted(pokemon.moves)))

    parse_log(record["battle_id"], record["spectator_log"], username, on_turn=on_turn)
    return traces


def plot_entropy(input_path: Path, output_path: Path) -> None:
    """Plot the largest observed first-to-last entropy drop in the local sample."""
    candidates = []
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            for perspective, username in sorted(record["players"].items()):
                for species, points in _role_traces(record, username).items():
                    if len(points) >= 2 and points[0][1] > points[-1][1]:
                        candidates.append((record["battle_id"], perspective, species, points))

    if not candidates:
        raise ValueError("No sampled opponent had a decrease in role entropy.")

    battle_id, perspective, species, points = max(
        candidates, key=lambda candidate: candidate[3][0][1] - candidate[3][-1][1]
    )
    turns, entropy, moves = zip(*points, strict=True)
    fig, axis = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    axis.scatter(turns, entropy, color="#2d6598", zorder=3)
    # Connect consecutive observed turns only. A gap means the Pokémon switched out.
    for first, second in zip(points, points[1:], strict=False):
        if second[0] == first[0] + 1:
            axis.plot((first[0], second[0]), (first[1], second[1]), color="#2d6598")
    axis.set(
        title=f"{species.title()} · Opponent role uncertainty",
        xlabel="Turn (start-of-turn observations while active)",
        ylabel="Role entropy (bits)",
        ylim=(-0.08, max(entropy) * 1.15),
    )
    axis.grid(axis="y", alpha=0.23)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"{battle_id}, observer {perspective}, opponent {species}")
    for turn, value, revealed in zip(turns, entropy, moves, strict=True):
        print(f"Turn {turn}: {value:.6f} bits ({', '.join(revealed) or 'no moves revealed'})")


def main() -> None:
    """Read a local spectator collection and save the illustrative entropy plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INTERIM_DIR / "local_games.jsonl")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "figures" / "belief_entropy.png")
    args = parser.parse_args()
    plot_entropy(args.input, args.out)


if __name__ == "__main__":
    main()
