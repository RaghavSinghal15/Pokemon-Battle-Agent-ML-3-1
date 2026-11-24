// Sample Gen 9 Random Battle teams from a local Showdown checkout and count, per species,
// how often each (moveset, ability, item) comes up. Used by pokeagent.beliefs.sample_items.
//   node scripts/sample_random_battle_items.js <showdown dir> <teams> <seed>
const path = require('path');
const [showdown, teamsArg, seedArg] = process.argv.slice(2);
const {Teams} = require(path.join(showdown, 'dist', 'sim'));
const toID = s => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '');
const counts = {};
// A fixed seed makes the sample, and so the item beliefs, the same on every machine.
const generator = Teams.getGenerator('gen9randombattle', [0, 0, 0, Number(seedArg)]);
for (let i = 0; i < Number(teamsArg); i++) {
  for (const set of generator.getTeam()) {
    const key = [toID(set.species), set.moves.map(toID).sort().join(','), toID(set.ability), toID(set.item)].join('|');
    counts[key] = (counts[key] || 0) + 1;
  }
}
process.stdout.write(JSON.stringify(counts));
