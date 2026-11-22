# Version 1 opponent-model features

The model predicts the opponent's next action from the start-of-turn public state.
In the table, `my_` means the observer's side and `opp_` means the side whose action
the model predicts. The same ordered columns are available in replay and live play.

| Feature | What it measures |
|---|---|
| `turn` | Number of the upcoming turn. |
| `my_hp` | Our active Pokémon's public HP fraction. |
| `opp_hp` | The opponent's active Pokémon's public HP fraction. |
| `my_remaining` | How many of our Pokémon are still alive, including unrevealed teammates. |
| `opp_remaining` | How many of their Pokémon are still alive, including unrevealed teammates. |
| `my_unrevealed` | How many of our Pokémon have yet to appear. |
| `opp_unrevealed` | How many of their Pokémon have yet to appear. |
| `my_boost_sum` | Sum of our active Pokémon's Attack, Defense, Special Attack, Special Defense and Speed stages. |
| `opp_boost_sum` | Sum of those five stages for their active Pokémon. |
| `my_status` | Whether our active Pokémon has a persistent status such as burn or sleep. |
| `opp_status` | Whether their active Pokémon has a persistent status. |
| `my_paralyzed` | Whether our active Pokémon is paralyzed. |
| `opp_paralyzed` | Whether their active Pokémon is paralyzed. |
| `my_hazard_layers` | Total layers of Stealth Rock, Spikes, Toxic Spikes and Sticky Web on our side. |
| `opp_hazard_layers` | Total layers of those entry hazards on their side. |
| `my_tera_used` | Whether our side has spent its once-per-battle Terastallization. |
| `opp_tera_used` | Whether their side has spent its Terastallization. |
| `my_revealed_moves` | Number of moves publicly revealed by our active Pokémon. |
| `opp_revealed_moves` | Number of moves publicly revealed by their active Pokémon. |
| `my_type_matchup` | Best type-chart multiplier from our active Pokémon's own types into theirs. |
| `opp_type_matchup` | Best type-chart multiplier from their active Pokémon's own types into ours. |
| `speed_advantage` | 1 if we are estimated to move first at equal priority, −1 if they are, or 0 for a tie or missing species. |
| `trick_room` | Whether Trick Room reverses the usual speed order. |
| `opp_choice_item_known` | Whether their active Pokémon is publicly known to still hold Choice Band, Choice Scarf or Choice Specs. |
| `opp_best_setup_move_probability` | Highest inclusion probability among their moves that directly raise one of the user's five battle stats. |
| `opp_best_recovery_move_probability` | Highest inclusion probability among their moves flagged as self-targeted healing. |
| `opp_last_attack` | Whether their previous observed action was a Physical or Special move. |
| `opp_last_non_damaging` | Whether their previous observed action was a Status move. |
| `opp_last_switch` | Whether their previous observed action was a switch. |
| `opp_expected_status_moves` | Expected number of Status (non-damaging) moves in their set: the sum of their inclusion probabilities. |
| `opp_status_move_useful` | Highest inclusion probability among their status-inflicting moves that could affect our active Pokémon now: it has no status and is not immune (Electric can't be paralysed, Fire can't be burned, Poison and Steel can't be poisoned, Grass ignores powder moves, and type immunities such as Ground against Thunder Wave). |
| `opp_hazard_move_useful` | Highest inclusion probability among their hazard moves (Stealth Rock, Spikes, Toxic Spikes, Sticky Web) whose hazard is not yet at its maximum on our side, if we have more than one Pokémon left. |
| `opp_hazard_removal_useful` | Highest inclusion probability among their hazard-removal moves (Rapid Spin, Defog, Mortal Spin, Tidy Up, Court Change), if hazards are on their side. |
| `opp_protect_probability` | Highest inclusion probability among their Protect-like moves. |
| `opp_substitute_probability` | Inclusion probability of Substitute. |
| `opp_pivot_move_probability` | Highest inclusion probability among moves that switch the user out (U-turn, Volt Switch, Flip Turn, Parting Shot, Teleport, Chilly Reception, Shed Tail). |

Type matchup is a proxy for same-type attacks. It uses the defending Tera type where
known, adds the attacking Tera type to the original types, and treats Stellar as the
original types for this approximation. Missing species give a neutral multiplier of 1.
The estimate ignores coverage moves, power and ability effects. Speed uses the species,
level, standard Random Battle spread, visible Speed stages, paralysis, a revealed Choice
Scarf and Trick Room. Other items, abilities, Tailwind and move priority can change the
actual turn order.

Missing active HP, boosts, status and move counts give zero. A missing or unknown previous
action sets all three action flags to zero. Missing beliefs give zero for both move
probability features. Taking the maximum of individual move probabilities gives the
probability of the single most likely move in each group. Those values do not measure the
chance of carrying at least one move in the group.

The current choice, effects later in the turn and the battle result are excluded.
## Export replay features

From the repository root, run `python -m pokeagent.features.export` after roles B
and C have produced `data/splits.json` and `data/processed/decision_points.parquet`.
The exporter reads the architecture's `state` JSON column and an `action` stored as
JSON or a Parquet struct. It writes `data/processed/features.parquet` with one row
per labelled decision point. It never recalculates the split.

Each input point uses the labelled player's perspective. The exporter swaps sides
and both previous actions before building features, so `opp_` describes the player
whose action is the label. The output `perspective` field still identifies that
labelled player, matching the input key. Train using the exact columns listed in
`FEATURE_NAMES`. Use `action_class` as the label and `split` to select rows.

## Value-model features (V1)

The value model predicts whether a player wins from the start-of-turn position. Its table,
`data/processed/value_features.parquet`, is written by `python -m pokeagent.features.value`.
It has one row per decision point and the label `won`.

The orientation is different from the opponent-model table. Here `my_` is the player whose
win we predict (the decision point's own perspective) and `opp_` is its opponent. So the
`opp_` belief features describe what that player's opponent might do to it.

The columns are all of `FEATURE_NAMES` above, plus eight whole-team features:

| Feature | What it measures |
|---|---|
| `my_team_hp` | Our team's total HP in Pokémon: revealed, unfainted HP fractions plus 1 per unrevealed teammate (they have not been hit). |
| `opp_team_hp` | The same for their team. |
| `my_statused` | How many of our revealed, unfainted Pokémon have a persistent status. |
| `opp_statused` | The same for their team. |
| `my_faster_than` | How many of their revealed, unfainted Pokémon our active Pokémon moves before (reversed under Trick Room). |
| `opp_faster_than` | How many of ours their active Pokémon moves before. |
| `my_super_effective` | How many of their revealed, unfainted Pokémon our active Pokémon's own types hit for 2× or more. |
| `opp_super_effective` | How many of ours their active Pokémon's types hit for 2× or more. |

Ladder ratings are left out on purpose. They predict the winner well, but they describe the
players, not the position. The agent also does not reliably see them during a live game.
