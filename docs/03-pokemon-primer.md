# Pokémon primer for people who have never played

This is the minimum game knowledge for the project. Roles A, B and E need little more than
sections 1 and 2. Roles C and F should know all of it. Role D should know it all and play a lot.

The best way to learn is to play: <https://play.pokemonshowdown.com>, choose
**"[Gen 9] Random Battle"**, click **Battle!**. No account needed. Play 10 to 15 games and you
will understand most of this page.

## 1. A battle, turn by turn

- Each player has **6 Pokémon**. One from each side is **active** (on the field) at a time.
- Each turn, **both players choose simultaneously**:
  - **use a move**: one of the active Pokémon's 4 moves, or
  - **switch**: bring in a teammate. The switch happens before attacks, so the incoming
    Pokémon takes the opponent's attack instead.
- A Pokémon at 0 HP (health) **faints**. Its player then picks a replacement (a "forced
  switch", which is not a simultaneous choice).
- You lose when all 6 of your Pokémon have fainted.

**Who moves first?** Switches happen first. Then moves go in order of **priority** (a few moves
like Quick Attack always go first), and within the same priority the Pokémon with the higher
**Speed** stat goes first. Under the field effect **Trick Room**, slower goes first. Moving first
matters a lot: if you knock the opponent out first, its attack never happens.

## 2. Stats, moves and damage

Every Pokémon has six stats: **HP**, **Attack**, **Defense**, **Special Attack**,
**Special Defense** and **Speed**.

Moves come in three **categories**:

| Category | Uses | Example |
|---|---|---|
| Physical | user's Attack vs target's Defense | Earthquake |
| Special | user's Special Attack vs target's Special Defense | Flamethrower |
| Status | deals no damage; does something else | Swords Dance (raises Attack) |

**Damage** depends on the move's power, the stats involved, type effectiveness, a random factor
between 85% and 100%, and many modifiers. Moves can miss (accuracy), and **critical hits**
(about 1 in 24) do 1.5x damage. We never compute damage by hand: poke-env has a damage
calculator.

## 3. Types: the rock-paper-scissors web

There are **18 types** (Normal, Fire, Water, Grass, Electric, Ice, Fighting, Poison, Ground,
Flying, Psychic, Bug, Rock, Ghost, Dragon, Dark, Steel, Fairy). Every Pokémon has one or two
types; every move has one type.

A move's damage is multiplied by how its type matches up against each of the target's types:

| Multiplier | Name | Example |
|---|---|---|
| 2x | super effective | Water vs Fire |
| 1x | neutral | Water vs Normal |
| 0.5x | not very effective | Water vs Grass |
| 0x | immune | Ground vs Flying |

Against a two-type target the multipliers multiply, so 4x and 0.25x happen. A Pokémon using a
move of its own type gets a 1.5x bonus (**STAB**, "same-type attack bonus"). Full chart:
<https://pokemondb.net/type>.

## 4. Things that change over a battle

- **Stat stages (boosts):** moves like Swords Dance raise a stat by stages, from -6 to +6
  (+1 = 1.5x, +2 = 2x). **Boosts disappear when the Pokémon switches out.** "Setting up" means
  spending turns boosting, then sweeping through the opponent's team.
- **Status conditions** (one at a time, they stay through switching):
  burn (loses HP each turn, physical damage halved), paralysis (Speed halved, sometimes can't
  move), sleep (can't move for a few turns), freeze (can't move until it thaws), poison and
  toxic (lose HP each turn; toxic gets worse every turn).
- **Entry hazards:** traps set on the *opponent's* side that damage every Pokémon they switch
  in: Stealth Rock, Spikes (up to 3 layers), Toxic Spikes, Sticky Web (lowers Speed). Moves like
  Rapid Spin and Defog remove them. Hazards punish switching.
- **Screens:** Reflect and Light Screen halve physical or special damage to one side for a few
  turns.
- **Weather and terrain:** rain, sun, sand, snow and four terrains boost or weaken certain types
  and moves for several turns.
- **Healing:** moves like Recover, items like Leftovers (heal a little every turn).

## 5. Abilities, items and terastallization

- **Ability:** a passive trait. Examples: Levitate (immune to Ground moves), Intimidate (lowers
  the opponent's Attack on switch-in).
- **Item:** each Pokémon holds one. Examples: Choice Scarf (1.5x Speed, but locks the Pokémon
  into the first move it uses until it switches out), Leftovers (heals every turn), Life Orb
  (more damage, costs HP), Heavy-Duty Boots (immune to hazards), Focus Sash (survives one hit
  from full HP).
- **Terastallization (tera):** once per battle, a player can change one Pokémon's type to its
  "tera type". This flips matchups: a Pokémon weak to Ground can suddenly resist it. It is a
  huge source of surprise.

## 6. What makes Random Battles special

- Both teams are **randomly generated** each game. There is no team preview: you only see an
  opponent's Pokémon when it comes onto the field.
- Weak species get higher **levels** and strong species lower levels, to balance teams.
- Each species has 1 to 3 **roles** (for example "Setup Sweeper" or "Fast Support"). A role
  defines a pool of moves (4 are picked), possible abilities and possible tera types. The item
  is chosen by the generator to fit. This is why we can reason about hidden information: once
  you see a Pokémon, only a few setups are possible, and every revealed move narrows them down.
- Stats are determined by species and level, so there is almost no hidden stat customisation.

## 7. How to read a battle log

Logs are text, one event per line, fields separated by `|`. `p1a` means player 1's active
Pokémon; HP is shown out of 100 to spectators.

```
|switch|p2a: Garchomp|Garchomp, L74|100/100    player 2 sends out a level-74 Garchomp
|turn|3                                         turn 3 begins: both players now choose
|move|p1a: Gholdengo|Make It Rain|p2a: Garchomp  player 1's Gholdengo uses Make It Rain
|-damage|p2a: Garchomp|61/100                    Garchomp drops to 61% HP
|-unboost|p1a: Gholdengo|spa|1                   Gholdengo's Special Attack falls one stage
|move|p2a: Garchomp|Earthquake|p1a: Gholdengo
|-supereffective|p1a: Gholdengo
|faint|p1a: Gholdengo                           Gholdengo faints
```

Lines starting with `-` are side effects of the action before them. The comments on the right
are ours; real logs don't have them.

To see real ones, open any game on <https://replay.pokemonshowdown.com> and add `.log` to the
end of its URL. Every event type is documented in `sim/SIM-PROTOCOL.md` in the
[pokemon-showdown repo](https://github.com/smogon/pokemon-showdown/blob/master/sim/SIM-PROTOCOL.md).

## 8. Where to look things up

- Any Pokémon, move, item or ability: <https://dex.pokemonshowdown.com>
- Damage calculator: <https://calc.pokemonshowdown.com>
- Random Battle roles and their odds: <https://pkmn.github.io/randbats/>
- The domain reviewer on the team.
