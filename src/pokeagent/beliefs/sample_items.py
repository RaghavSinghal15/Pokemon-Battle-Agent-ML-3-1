"""Sample Random Battle teams from the local simulator to estimate item beliefs (#119).

    python -m pokeagent.beliefs.sample_items            # RANDOM_BATTLE_ITEM_TEAMS teams

Needs Node.js and the built pokemon-showdown checkout that runs local battles (the same one
sets.json comes from). Writes RANDOM_BATTLE_ITEMS_PATH (data/, git-ignored, like all data).
"""

import argparse
import json
import subprocess

from pokeagent.config import (
    RANDOM_BATTLE_ITEM_TEAMS,
    RANDOM_BATTLE_ITEMS_PATH,
    RANDOM_BATTLE_SHOWDOWN_DIR,
    REPO_ROOT,
    SEED,
)

SCRIPT = REPO_ROOT / "scripts" / "sample_random_battle_items.js"


def main(teams: int = RANDOM_BATTLE_ITEM_TEAMS) -> None:
    output = subprocess.run(
        ["node", str(SCRIPT), str(RANDOM_BATTLE_SHOWDOWN_DIR), str(teams), str(SEED)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    counts = json.loads(output)
    RANDOM_BATTLE_ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RANDOM_BATTLE_ITEMS_PATH.write_text(json.dumps(counts), encoding="utf-8")
    species = {key.split("|")[0] for key in counts}
    print(f"{teams:,} teams, {len(species)} species -> {RANDOM_BATTLE_ITEMS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teams", type=int, default=RANDOM_BATTLE_ITEM_TEAMS)
    main(parser.parse_args().teams)
