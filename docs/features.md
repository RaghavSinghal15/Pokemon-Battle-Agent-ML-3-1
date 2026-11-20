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
Battle IDs and split names are table metadata. For the replay table, flip the perspective
of each labelled decision point so that its observed actor is the model's opponent.
