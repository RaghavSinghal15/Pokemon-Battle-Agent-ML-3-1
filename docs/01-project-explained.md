# The project, explained in plain language

## What we are building

An AI agent that plays **Pokémon Showdown**, a free online two-player strategy game, and
explains every decision it makes. We play the **Gen 9 Random Battles** format.

The game in one paragraph: each player has a team of six characters (Pokémon). One from each
side is on the field at a time. Every turn, **both players choose at the same time**: either use
one of their active Pokémon's four moves (usually an attack), or swap it out for a teammate.
Attacks reduce the target's health; at zero health it faints. The first player to have all six
of their Pokémon faint loses.

In **Random Battles**, both teams are generated randomly for every game, so nobody can rely on
a memorised team or strategy. Every game is a new puzzle.

## Why it is hard

There are two layers of difficulty.

**Layer 1: the game is huge.** 18 types (Fire, Water, Grass and so on) form a large
rock-paper-scissors web, so the same attack can do double damage, half damage, or nothing
depending on the target. On top of that come abilities, held items, moves that boost later
attacks, heal, poison, set traps on the opponent's side, change the weather, or disable the
opponent's moves; switching; and terastallizing (changing a Pokémon's type once per game).
These mechanics combine, so no fixed set of rules covers every situation.

**Layer 2: every decision is made under uncertainty.**

- **Simultaneous choices.** You commit without seeing what the opponent picked. Like
  rock-paper-scissors, the best move depends on theirs, so often there is no single best move.
- **Hidden information.** You don't know the opponent's moves, item, ability or remaining team
  until they are revealed. Like poker, you reason about what they probably have.
- **Randomness.** Damage varies, moves miss, critical hits happen.
- **Long-term consequences.** Good players sacrifice a Pokémon or spend turns setting up for a
  win later. The best-looking move now can lose the game.

This is why a bot that always uses its strongest-looking attack loses badly to average humans.

## How our agent thinks, in one picture

Every turn, a good human player does three things in their head. Our agent does the same:

1. **What does the opponent probably have?** (beliefs)
2. **What will the opponent probably do?** (opponent model, learned from human games)
3. **Given that, what is my best move?** (decision engine)

Everything else in the project exists to feed these three steps, or to measure how well they
work. Here is each piece of the pipeline, in the order data flows through it.

---

## Step A: The practice arena and the scoreboard (role A: infrastructure and evaluation)

Showdown is open source, so we run **our own private copy of the game server** on a laptop.
Bots can play each other thousands of times there, fast (several games per second between simple bots), without
touching the real website.

**poke-env** is a Python library that lets Python code act as a player: it receives the battle
as data and sends back choices like "use move 2" or "switch to the Pokémon in slot 4".

**Baselines** are simple bots we measure ourselves against:

- **Random:** picks any legal action.
- **Max power:** always uses the move with the highest listed power. This is roughly how a
  child plays.
- **Simple heuristics:** a hand-written rule-based bot. Decent, and the one to beat.

**Evaluation** means playing a few hundred games against each baseline, counting wins and
reporting the result with an error margin, for example "68% ± 4%".

*Plain version: build the gym and the scoreboard.*

## Step B: Getting the recorded games (role B: data pipeline)

A public dataset contains about 2 million recorded Random Battles between human players. Each
recording (a **replay**) is a text log where every line is one event. We download a part of it,
keep only games between reasonably strong players (we want to learn how good players behave),
remove duplicates, and store everything in **Parquet**, a compressed table format that loads
fast. Then we split the games into training, validation and test sets, and write a short data
report: how many games, the spread of player ratings, how long games last.

*Plain version: collect and clean the study material.*

## Step C: Rebuilding what was visible each turn (role C: parser)

A replay log looks like this:

```
|turn|3
|switch|p2a: Garchomp|Garchomp, L74|100/100
|move|p1a: Gholdengo|Make It Rain|p2a: Garchomp
|-damage|p2a: Garchomp|61/100
```

The parser walks through it line by line and builds a **snapshot** at the start of every turn:
which Pokémon are out, their health, everything revealed so far, and what is still unknown. It
also records what each player chose that turn. Each (snapshot, choice) pair is one training
example, called a **decision point**.

The key rule is **no leakage**: the snapshot for turn 3 must not contain anything revealed
later. If Garchomp only uses Earthquake on turn 10, the turn-3 snapshot must not know about it.
Otherwise the model looks brilliant in testing and fails in real games, like studying with the
answer key stapled to the question paper.

A replay is the **spectator's** view, so even a player's own unrevealed moves are not in the
log. We therefore build snapshots from **public information only, for both sides**, in training
and in live play, so the model always sees the same kind of input. (Details:
[04-architecture.md](04-architecture.md).)

We check the parser by playing games on our local server, where we can see everything, and
comparing its snapshots against the truth.

*Plain version: turn messy logs into clean "what was known at this moment" snapshots, without
cheating.*

## Step D: Guessing hidden information, and describing the situation in numbers (role D)

**Beliefs.** In Random Battles, each species can only be generated in one of a few **roles**
(usually 1 to 3), each with its own pool of possible moves, abilities and tera types. When the
opponent sends out Garchomp, we start with probabilities like "Setup Sweeper 55%, Fast Support
45%". Every time something is revealed, say it uses Spikes, which only the Fast Support role has,
we cross out the roles that could not have done that and rescale the rest so they add up to
100%. That is **Bayes' rule** in its simplest form. Think of the game Guess Who: every clue
eliminates options.

**Features.** A model cannot read a snapshot; it needs a row of numbers. So we turn each
snapshot into numbers like: our health, the opponent's health, the type-effectiveness
multiplier each way, who is probably faster, how many Pokémon each side has left, stat boosts,
status conditions, traps on the field, belief probabilities ("80% chance it has a setup move"),
and what the opponent did last turn. Choosing **which** numbers matter is where game knowledge
pays off most.

*Plain version: play detective on the opponent's team, then describe the situation in numbers.*

## Step E: Predicting the opponent (role E: opponent model)

A **classifier**, exactly like the ones in the course. Input: the feature row. Output:
probabilities for what the opponent does next. In version 1 we predict the *type* of action
rather than the exact move, because every Pokémon has different moves:

- **attack** (a damaging move),
- **non_damaging** (boosting, healing, setting traps, inflicting status),
- **switch**.

We compare **multinomial logistic regression** (simple, interpretable, course syllabus) with
**LightGBM** (gradient-boosted decision trees, usually stronger; course topic: ensembles).

Then we **calibrate**: when the model says 70%, the event should happen about 70% of the time,
like a weather forecaster whose "70% rain" days really are rainy 7 times out of 10. This matters
because the next step multiplies by these numbers. Overconfident probabilities make bad bets.

*Plain version: learn from human games how people tend to play in each situation.*

## Step F: Choosing a move (role F: decision engine)

Build a grid, the **payoff table**:

- rows: our options (up to 4 moves, plus switches),
- columns: the opponent's likely options,
- each cell: how good that combination is for us, computed with a **damage calculator** (how
  much health each side loses, whether anything faints, who moves first).

Then pick a move in one of two ways:

- **Best response:** weight each column by how likely the opponent model says it is, and pick
  the row with the best average. *Play the odds against how this person probably plays.*
- **Safe mixed strategy (minimax):** assume the opponent plays perfectly against us, and find
  the mix of moves with the best *guaranteed* result, for example "Earthquake 60% of the time,
  switch 40%". In rock-paper-scissors this gives one third each. **Linear programming** finds
  this mix exactly.

We then test which strategy wins more, and against which opponents. This comparison is one of
the main results of the project.

*Plain version: weigh every "if I do X and they do Y" and choose wisely.*

## Step G: Explaining itself (roles F and E)

Every turn, the agent can print what it believes about the opponent's Pokémon, what it predicts
the opponent will do, the payoff table, and why it chose its move. This is what makes it
**explainable**, and it is the centrepiece of the demo.

---

## What makes this a strong ML project (for the report and the viva)

- **Problem formulation**, not "method on a dataset": we turn a game with hidden information
  and simultaneous moves into a prediction problem plus a decision problem.
- **A full pipeline**: data collection, cleaning, state reconstruction, features, training,
  calibration, decision-making, evaluation.
- **Course methods used for a real reason**: logistic regression, ensembles (boosted trees),
  Bayes' rule, calibration, evaluation with confidence intervals, plus linear programming.
- **Analysis**: logistic regression vs boosted trees, best response vs minimax, and ablations
  (turn off beliefs, or the opponent model, and measure how much worse the agent gets).
- **Explainability**: every decision can be traced and explained.
- **Prior work we can compare with**: showdown-brain (an open-source Random Battles bot with a
  learned win predictor) and Metamon (UT Austin, offline reinforcement learning on the
  Overused format).
