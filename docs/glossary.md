# Glossary

## Game terms

| Term | Meaning |
|---|---|
| Active | The Pokémon currently on the field for a side |
| Ability | A Pokémon's passive trait (e.g. Levitate: immune to Ground moves) |
| Boost / stat stage | A temporary stat change from -6 to +6; resets on switching out |
| Choice item | Choice Scarf / Band / Specs: big boost, but locks the holder into one move |
| Critical hit (crit) | A random 1.5x damage hit (about 1 in 24) |
| Entry hazard | A trap on one side that damages Pokémon switching in (Stealth Rock, Spikes...) |
| Faint | Reaching 0 HP; the Pokémon is out for the rest of the game |
| Forced switch | Choosing a replacement after a faint; not a simultaneous choice |
| HP | Health points; shown to spectators as a percentage |
| Item | The one object a Pokémon holds (Leftovers, Life Orb...) |
| Ladder | Showdown's public ranked matchmaking, with ratings |
| Level | Random Battles give weaker species higher levels to balance teams |
| Move | One of a Pokémon's 4 actions; physical, special or status |
| Pivot | A move that attacks then switches out (U-turn, Volt Switch) |
| Priority | Some moves always go first regardless of Speed |
| Random Battle | Format where both teams are randomly generated every game |
| Replay | A saved recording of a game, as a text log of events |
| Role | In Random Battles, one of a species' 1 to 3 possible builds (move pool, abilities, tera types) |
| Screens | Reflect / Light Screen: halve physical / special damage to a side for a few turns |
| Set | The full build of one Pokémon: moves, item, ability, tera type |
| Setup | Spending turns boosting stats before attacking |
| STAB | Same-type attack bonus: 1.5x for moves matching the user's type |
| Status condition | Burn, paralysis, sleep, freeze, poison, toxic |
| Switch | Replacing the active Pokémon with a teammate; costs the turn |
| Tera / terastallize | Once per battle, change one Pokémon's type to its tera type |
| Type effectiveness | The 0x / 0.5x / 1x / 2x damage multiplier between move and target types |
| Weather / terrain | Field effects that boost or weaken certain types and moves |

## ML and project terms

| Term | Meaning |
|---|---|
| Ablation | Removing one component and measuring how much performance drops |
| Action class | Our label for the opponent model: attack, non_damaging or switch |
| Baseline | A simple method we compare against (random bot, max power bot, heuristics bot, class-frequency predictor) |
| Bayes' rule | P(A given B) is proportional to P(B given A) × P(A); used to update beliefs |
| Belief | Our probability distribution over an opponent Pokémon's hidden details |
| Best response | Choosing the action with the highest expected payoff given predicted opponent probabilities |
| Calibration | Making predicted probabilities match observed frequencies |
| Confidence interval | A range built so that, if the experiment were repeated many times, 95% of such ranges would contain the true value (e.g. the true win rate) |
| Decision point | One training example: a state and the action a player chose |
| Expected value | Probability-weighted average of outcomes |
| Feature | One number describing a state, used as model input |
| Isotonic regression / Platt scaling | Two calibration methods (flexible step function vs fitted sigmoid) |
| Leakage | Information from the future or from the label getting into model inputs |
| LightGBM | A fast gradient-boosted decision tree library |
| Linear programming (LP) | Optimising a linear objective subject to linear constraints; solves the minimax strategy |
| Log loss | Penalises confident wrong probabilities; the standard metric for probabilistic predictions |
| Macro-F1 | F1 score averaged equally over classes, so rare classes count |
| Minimax / mixed strategy | A randomised strategy maximising the worst-case expected payoff |
| Multinomial logistic regression | Logistic regression for more than two classes (softmax output) |
| Parquet | A compressed, column-based table file format |
| Payoff table | Grid of how good each (our action, their action) pair is for us |
| poke-env | Python library for writing Showdown bots |
| Public state | What a spectator can see at the start of a turn |
| Walking skeleton | A system that runs end to end from day one, with placeholder parts replaced over time |
