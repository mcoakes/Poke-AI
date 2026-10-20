'use strict';
// Private, pinned simulator API. No server, sockets, or GUI is started.
const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const {Battle, Dex, toID} = require('pokemon-showdown');
const version = require('pokemon-showdown/package.json').version;
if (version !== '0.11.11') throw new Error('Run npm ci in sim/showdown (requires pokemon-showdown 0.11.11)');
const root = path.resolve(__dirname, '../..');
const names = require(path.join(root, 'mgba/profiles/firered_us_v1_0_names.json'));
const catalog = require(path.join(root, 'config/generation/firered_us_v1_0.json'));
const dex = Dex.mod('gen3');
const key = x => toID(String(x).replaceAll('♀', 'f').replaceAll('♂', 'm'));
const ids = Object.fromEntries(Object.entries(names).map(([group, entries]) =>
  [group, Object.fromEntries(Object.entries(entries).map(([id, name]) => [key(name), Number(id)]))]));
ids.types['???'] = 9;
const allowed = Object.fromEntries(['species', 'moves', 'items'].map(group =>
  [group, new Set(Object.keys(catalog[group]).map(key))]));
// Some cartridge spellings differ (e.g. Smelling Salt / Smelling Salts).
// Move numbers are stable across the two data sources.
const cartridgeMoves = Object.fromEntries(Object.entries(catalog.moves).map(([name, data]) => {
  let move = dex.moves.get(name);
  if (!move.exists || move.num !== data.id) move = dex.moves.all().find(m => m.num === data.id && m.gen <= 3);
  if (!move) throw new Error(`No simulator mapping for cartridge move ${name} (${data.id})`);
  return [key(name), move];
}));
const natures = ['Hardy','Lonely','Brave','Adamant','Naughty','Bold','Docile','Relaxed','Impish','Lax',
  'Timid','Hasty','Serious','Jolly','Naive','Modest','Mild','Quiet','Bashful','Rash','Calm','Gentle','Sassy','Careful','Quirky'];
const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
let battle, original, logOffset = 0;

function sets(team) {
  if (!Array.isArray(team.party) || team.party.length < 1 || team.party.length > 6) throw new Error('Team size must be 1–6');
  return team.party.map((p, i) => {
    if (!allowed.species.has(key(p.species))) throw new Error(`Unknown Gen III species: ${p.species}`);
    const species = dex.species.get(key(p.species) === 'deoxys' ? 'Deoxys-Attack' : p.species);
    if (!species.exists || species.gen > 3) throw new Error(`Unsupported species: ${p.species}`);
    if (!Array.isArray(p.moves) || p.moves.length < 1 || p.moves.length > 4) throw new Error('Provide 1–4 moves');
    const moves = p.moves.map(name => {
      if (!allowed.moves.has(key(name))) throw new Error(`Unknown Gen III move: ${name}`);
      const move = cartridgeMoves[key(name)];
      if (!move.exists || move.gen > 3) throw new Error(`Unsupported move: ${name}`);
      return move.id;
    });
    if (new Set(moves).size !== moves.length) throw new Error('Duplicate moves');
    if (!Number.isInteger(p.level) || p.level < 1 || p.level > 100) throw new Error('Level must be 1–100');
    const item = p.held_item || 'None';
    if (!allowed.items.has(key(item))) throw new Error(`Unknown Gen III item: ${item}`);
    // Match TeamPokemonBuilder: first ability bit; PID-derived nature/gender; friendship 255.
    let pid = (0x13579bdf + i * 0x10201) >>> 0;
    const nature = p.nature ? natures.findIndex(n => key(n) === key(p.nature)) : pid % 25;
    if (nature < 0) throw new Error(`Unknown nature: ${p.nature}`);
    while (pid % 25 !== nature) pid = (pid + 2) >>> 0;
    const gender = species.gender || ((pid & 255) < Math.round(species.genderRatio.F * 256)-1 ? 'F' : 'M');
    const evs = Object.fromEntries(stats.map(s => [s, p.evs?.[s] ?? 0]));
    const ivs = Object.fromEntries(stats.map(s => [s, p.ivs?.[s] ?? 31]));
    if (Object.values(evs).some(v => !Number.isInteger(v) || v < 0 || v > 255) ||
        Object.values(evs).reduce((a, b) => a + b, 0) > 510 ||
        Object.values(ivs).some(v => !Number.isInteger(v) || v < 0 || v > 31)) throw new Error('Invalid EVs/IVs');
    return {name: p.species.toUpperCase(), species: species.name, moves, level: p.level,
      ability: species.abilities[0], item: key(item) === 'none' ? '' : item,
      evs, ivs, nature: natures[nature], gender, happiness: 255};
  });
}

function typeIds(types) {
  const values = types.map(t => t === '???' ? 9 : ids.types[key(t)]);
  if (values.some(t => t === undefined)) throw new Error(`Non-Gen-III types: ${types}`);
  return [values[0], values[1] ?? values[0]];
}
function status(p) {
  if (p.status === 'slp') return Math.max(1, Math.min(7, p.statusState.time || 1));
  if (p.status === 'tox') return 128 | (Math.min(15, p.statusState.stage || 0) << 8);
  return {psn:8, brn:16, frz:32, par:64}[p.status] || 0;
}
function volatile(p) {
  const v = p.volatiles;
  let raw = 0;
  // Confusion counts attempted moves in time, not residual-effect duration.
  if (v.confusion) raw |= Math.max(1, Math.min(7, v.confusion.time || 1));
  for (const [name, shift, limit] of [['uproar',4,7],['bide',8,3],
      ['lockedmove',10,3],['partiallytrapped',13,7]]) {
    if (v[name]) raw |= Math.max(1, Math.min(limit, v[name].duration || 1)) << shift;
  }
  for (const [name, bit] of Object.entries({flinch:3, focusenergy:20, mustrecharge:22,
    rage:23, substitute:24, destinybond:25, trapped:26, nightmare:27, curse:28,
    foresight:29, defensecurl:30, torment:31})) if (v[name]) raw |= 1 << bit;
  if (p.transformed) raw |= 1 << 21;
  if (v.twoturnmove || v.lockedmove || v.bide || v.uproar || v.rollout || v.iceball) raw |= 1 << 12;
  if (v.attract) raw |= 1 << (16 + (1 - p.side.n));
  return raw >>> 0;
}
function moves(slots) {
  return slots.map((m, i) => {
    const data = dex.moves.get(m.id);
    return {slot:i+1, move_id:data.num, type_id:typeIds([data.type])[0],
      base_type_id:typeIds([data.type])[0], current_pp:m.pp, max_pp:m.maxpp, power:data.basePower};
  });
}
function mon(p, active = false) {
  // Party ordering never changes in the policy, even though Showdown swaps its array on switch.
  const slot = original[p.side.n].indexOf(p) + 1;
  const species = active ? p.species : p.baseSpecies;
  const speciesId = ids.species[key(species.baseSpecies)];
  if (speciesId === undefined) throw new Error(`Unmapped FireRed species ID: ${species.name}`);
  return {slot, party_slot:slot, valid:true, occupied:true, is_egg:false,
    species_id:speciesId, level:p.level, current_hp:p.hp, max_hp:p.maxhp,
    base_type_ids:typeIds(p.baseSpecies.types), type_ids:typeIds(active ? p.getTypes() : p.baseSpecies.types),
    status_raw:status(p), volatile_status_raw:active ? volatile(p) : 0,
    held_item_id:p.item ? (ids.items[key(p.item)] ?? ids.items[key(p.getItem().name)] ?? 0) : 0,
    ability_id:p.getAbility().num || 0, moves:moves(active ? p.moveSlots : p.baseMoveSlots)};
}

const battleStats = ['atk', 'def', 'spa', 'spd', 'spe'];
const boostStats = [...battleStats, 'accuracy', 'evasion'];
const specialTypes = new Set(['Fire', 'Water', 'Grass', 'Ice', 'Electric', 'Dark', 'Psychic', 'Dragon']);
function itemId(p) {
  // Cartridge item numbers differ from modern Showdown item numbers.
  return p.item ? (ids.items[key(p.item)] ?? ids.items[key(dex.items.get(p.item).name)] ?? 0) : 0;
}
function abilityId(p) {
  // Natural Gen III ability numbers (1–77); zero denotes no ability.
  return p.ability ? dex.abilities.get(p.ability).num || 0 : 0;
}
function moveDetails(p, slots, active) {
  return Array.from({length:4}, (_, index) => {
    const slot = slots[index];
    if (!slot) return [0, 0, 0, 0, 0, 0];
    const move = dex.moves.get(slot.id);
    const hiddenPower = move.id === 'hiddenpower';
    // These are base descriptors, not damage predictions: variable power,
    // weather-modified accuracy, and other contextual callbacks are not run.
    const type = hiddenPower ? (active ? p.hpType : p.baseHpType) : move.type;
    const power = hiddenPower ? (active ? p.hpPower : p.baseHpPower) : move.basePower;
    const category = hiddenPower ? (specialTypes.has(type) ? 1 : 0) :
      {Physical:0, Special:1, Status:2}[move.category];
    return [power, move.accuracy === true ? 0 : move.accuracy,
      Number(move.accuracy === true), category, move.priority, typeIds([type])[0]];
  });
}
function observationV2(perspective) {
  // Read native stored facts only. Pokemon getters may trigger battle events
  // or randomness; neither observations nor audit inspection may advance RNG.
  const sides = [perspective, 1 - perspective];
  const party = sides.map(side => Array.from({length:6}, (_, slot) => {
    const p = original[side][slot];
    return p ? [1, p.level, p.hp, p.maxhp, ...typeIds(p.baseSpecies.types),
      itemId(p), abilityId(p), Number(p.fainted), status(p)] : Array(10).fill(0);
  }));
  const partyStats = sides.map(side => Array.from({length:6}, (_, slot) => {
    const p = original[side][slot];
    return p ? battleStats.map(stat => p.baseStoredStats[stat]) : Array(5).fill(0);
  }));
  const active = sides.map(side => battle.sides[side].active[0]);
  const details = sides.map(side => Array.from({length:6}, (_, slot) => {
    const p = original[side][slot];
    // Replace the entire active row after Transform/Mimic, including its empty
    // slots, so a shorter copied moveset cannot leak the original trailing moves.
    const isActive = p === battle.sides[side].active[0];
    return p ? moveDetails(p, isActive ? p.moveSlots : p.baseMoveSlots, isActive) :
      Array.from({length:4}, () => Array(6).fill(0));
  }));
  const weather = {sunnyday:1, raindance:2, sandstorm:3, hail:4}[battle.field.weather] || 0;
  const duration = weather ? battle.field.weatherState.duration || 0 : 0;
  return {v2_party:party, v2_stats:partyStats,
    v2_active_stats:active.map(p => p ? battleStats.map(stat => p.storedStats[stat]) : Array(5).fill(0)),
    v2_active:active.map(p => p ? [p.hp, p.maxhp, itemId(p), abilityId(p)] : Array(4).fill(0)),
    v2_boosts:active.map(p => p ? boostStats.map(stat => p.boosts[stat]) : Array(7).fill(0)),
    v2_move_details:details, v2_weather:[weather, duration, Number(!!weather && !duration)],
    v2_volatiles:active.map(p => p ? volatile(p) : 0)};
}

function choices(side) {
  const request = side.activeRequest;
  const mask = Array(11).fill(0), commands = Array(11).fill(null);
  if (battle.ended) return {mask, commands, forced:false, automatic:false};
  if (!request || request.wait) { mask[10] = 1; return {mask, commands, forced:false, automatic:true}; }
  if (request.teamPreview) throw new Error('Team preview is outside the FireRed policy contract');
  const p = side.active[0];
  const forced = !!request.forceSwitch?.[0];
  if (!forced) {
    // Private/full-state requests avoid hiding Disable/trapping that our mGBA mask knows.
    const hardLock = p.getLockedMove();
    const locked = hardLock || p.getSemiLockedMove(false);
    const available = p.getMoves(locked, false);
    if (hardLock) {
      mask[10] = 1; commands[10] = 'move 1';
      return {mask, commands, forced:false, automatic:true};
    }
    if (!available.length) {mask[0] = 1; commands[0] = 'move 1';}
    else for (const m of available) {
      const logical = p.moveSlots.findIndex(s => s.id === m.id);
      const requestIndex = request.active[0].moves.findIndex(s => s.id === m.id);
      if (!m.disabled && (m.pp === undefined || m.pp > 0) && logical >= 0 && requestIndex >= 0) {
        mask[logical] = 1; commands[logical] = `move ${requestIndex+1}`;
      }
    }
  }
  if (forced || !p.trapped) {
    for (const other of original[side.n]) if (other.hp > 0 && other !== p && !other.fainted) {
      const action = 4 + original[side.n].indexOf(other);
      mask[action] = 1; commands[action] = `switch ${side.pokemon.indexOf(other)+1}`;
    }
  }
  if (!mask.some(Boolean)) throw new Error(`No legal actions for ${side.id}: ${JSON.stringify(request)}`);
  return {mask, commands, forced, automatic:false};
}

function snapshot() {
  const decisions = battle.sides.map(choices);
  const reports = battle.sides.map((side, i) => {
    const outcome = !battle.ended ? 'ongoing' : !battle.winner ? 'draw' : battle.winner === side.name ? 'won' : 'lost';
    const party = original[i].map(p => mon(p));
    const opponent = original[1-i].map(p => mon(p));
    const counts = records => ({valid:true, remaining:records.filter(p => p.current_hp > 0).length,
      fainted:records.filter(p => p.current_hp === 0).length});
    return {ok:true, profile:'firered_us_v1_0', backend:'showdown', frame:0, party, bag:{},
      observation_v2:observationV2(i),
      action_mask:decisions[i].mask, battle:{active:true, valid:true, turn:Math.max(0, battle.turn-1),
        type_flags:10, outcome, outcome_raw:{ongoing:0, won:1, lost:2, draw:3}[outcome],
        terminal:battle.ended, player_counts:counts(party), opponent_counts:counts(opponent),
        opponent_party:opponent, battlers:battle.sides.map((s, j) => ({...mon(s.active[0], true),
          side:j === i ? 'player' : 'opponent', battler_id:j, absent:false})),
        menus:[{state:decisions[i].forced ? 'party' : decisions[i].automatic ? 'busy_or_unknown' : 'command',
          submenu:decisions[i].forced ? 'choose_pokemon' : undefined, switch_required:decisions[i].forced}]}};
  });
  const protocol = battle.log.slice(logOffset);
  logOffset = battle.log.length;
  return {ok:true, version, reports, protocol, ended:battle.ended};
}

function submit(actions, captureExecution = false) {
  const options = battle.sides.map(choices);
  if (!Array.isArray(actions) || actions.length !== 2 || actions.some((a,i) =>
      !Number.isInteger(a) || !options[i].mask[a])) throw new Error(`Illegal paired actions: ${actions}`);
  // Optional diagnostic evidence, captured before either side acts. Stored move
  // slot 1 can still say Rock Slide while the native request offers Struggle.
  const execution = captureExecution ? battle.sides.map((side, i) => {
    const command = options[i].commands[actions[i]];
    const moveIndex = /^move (\d+)$/.exec(command || '');
    return {action:actions[i], command,
      requested_move:moveIndex ? side.activeRequest?.active?.[0]?.moves[Number(moveIndex[1])-1]?.id ?? null : null,
      resolved_move:null};
  }) : null;
  // Validate and buffer both choices before executing either player's move.
  for (let i = 0; i < 2; i++) {
    const command = options[i].commands[actions[i]];
    if (command && !battle.sides[i].choose(command)) throw new Error(battle.sides[i].choice.error || `Rejected ${command}`);
    if (execution) execution[i].resolved_move = battle.sides[i].choice.actions.find(
      a => a.choice === 'move' && a.pokemon === battle.sides[i].active[0])?.moveid ?? null;
  }
  if (!battle.allChoicesDone()) throw new Error('Incomplete simultaneous choices');
  battle.commitChoices();
  return execution;
}

function handle(request) {
  if (request.command === 'ping') return {ok:true, version, generation:3};
  if (request.command === 'reset') {
    if (!Array.isArray(request.teams) || request.teams.length !== 2) throw new Error('Need two teams');
    const teams = request.teams.map(sets);
    if (battle) battle.destroy();
    battle = new Battle({formatid:'gen3customgame', seed:request.seed, strictChoices:true});
    battle.setPlayer('p1', {name:'Player 1', team:teams[0]});
    battle.setPlayer('p2', {name:'Player 2', team:teams[1]});
    original = battle.sides.map(side => side.pokemon.slice());
    for (const party of original) for (const p of party) {
      p.ppUps.fill(0);
      for (const list of [p.baseMoveSlots, p.moveSlots]) for (const move of list) {
        move.pp = move.maxpp = battle.calculatePP(dex.moves.get(move.id), 0);
      }
    }
    battle.makeRequest('move');
    logOffset = 0;
    return snapshot();
  }
  if (request.command === 'step') {
    if (!battle || battle.ended) throw new Error('Reset before stepping');
    const execution = submit(request.actions, request.audit_execution === true);
    // No policy choice exists when both sides are recharging/locked/waiting.
    while (!battle.ended && battle.turn < 256 && battle.sides.map(choices).every(c => c.automatic)) submit([10,10]);
    const response = snapshot();
    if (execution) response.audit_execution = execution;
    return response;
  }
  throw new Error(`Unknown command: ${request.command}`);
}

if (require.main === module) {
  const input = readline.createInterface({input:process.stdin, crlfDelay:Infinity});
  input.on('line', line => {
    try {process.stdout.write(JSON.stringify(handle(JSON.parse(line))) + '\n');}
    catch (error) {process.stdout.write(JSON.stringify({ok:false, error:error.message, stack:error.stack}) + '\n');}
  });
  input.on('close', () => {if (battle) battle.destroy();});
}
// Private read-only inspection hook for audit_runner.cjs. It deliberately exposes
// native objects rather than calling the adapter's report or legality helpers.
// The production NDJSON protocol remains limited to ping/reset/step.
function auditSource() { return {battle, original, version, logOffset}; }
module.exports = {handle, sets, auditSource};
