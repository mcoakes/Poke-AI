'use strict';
// Audit-only companion to the pinned production worker. All state transitions
// use its ordinary handle(); inspection reads native Showdown state directly.
// Never call Pokemon event-backed getters here: even a getter can run handlers,
// mutate request state, or consume randomness. The Python oracle interprets the
// native request, slot state, and volatile facts independently.
const readline = require('node:readline');
const production = require('./runner.cjs');

function inspect() {
  const {battle, original, version, logOffset} = production.auditSource();
  if (!battle) throw new Error('Reset before inspecting');
  const slots = new Map();
  original.forEach((party, side) => party.forEach((pokemon, index) => {
    slots.set(pokemon, {side, slot:index + 1});
  }));
  // Effect states contain references back to Pokemon/Battle. Preserve useful
  // references as stable identities while copying every plain native field.
  function copy(value, seen = new Set()) {
    if (value === null || typeof value !== 'object') {
      return typeof value === 'function' || value === undefined ? undefined : value;
    }
    if (slots.has(value)) return {pokemon: {...slots.get(value)}};
    if (value === battle) return {battle:true};
    if (battle.sides.includes(value)) return {side:value.n};
    if (seen.has(value)) return {circular:true};
    const next = new Set(seen).add(value);
    if (Array.isArray(value)) return value.map(item => copy(item, next));
    if (value instanceof Set) return [...value].map(item => copy(item, next));
    if (value instanceof Map) return [...value].map(item => copy(item, next));
    const result = {};
    for (const [name, item] of Object.entries(value)) {
      const copied = copy(item, next);
      if (copied !== undefined) result[name] = copied;
    }
    return result;
  }
  const species = value => ({id:value.id, name:value.name, num:value.num,
    base_species:value.baseSpecies, types:value.types.slice(),
    base_stats:{...value.baseStats}, abilities:{...value.abilities}});
  const moveSlots = values => values.map((value, index) => {
    const data = battle.dex.moves.get(value.id);
    return {slot:index + 1, id:value.id, name:data.name, num:data.num,
      type:data.type, power:data.basePower, category:data.category,
      accuracy:data.accuracy, priority:data.priority,
      pp:value.pp, maxpp:value.maxpp, base_pp:data.pp, target:value.target,
      disabled:value.disabled, disabled_source:value.disabledSource, used:value.used};
  });
  function pokemon(value) {
    const item = battle.dex.items.get(value.item);
    const ability = battle.dex.abilities.get(value.ability);
    return {slot:slots.get(value).slot, position:value.position,
      active:value.isActive, fainted:value.fainted, faint_queued:value.faintQueued,
      name:value.name, species:species(value.species), base_species:species(value.baseSpecies),
      level:value.level, hp:value.hp, maxhp:value.maxhp, base_maxhp:value.baseMaxhp,
      status:value.status, status_state:copy(value.statusState),
      volatiles:copy(value.volatiles), types:value.types.slice(),
      base_types:value.baseTypes.slice(), added_type:value.addedType,
      transformed:value.transformed,
      item:{id:value.item, name:item.name, num:item.num},
      ability:{id:value.ability, name:ability.name, num:ability.num},
      base_ability:value.baseAbility, item_state:copy(value.itemState),
      ability_state:copy(value.abilityState),
      stored_stats:{...value.storedStats}, base_stored_stats:copy(value.baseStoredStats),
      boosts:{...value.boosts}, speed:value.speed,
      set:copy(value.set), moves:moveSlots(value.moveSlots), base_moves:moveSlots(value.baseMoveSlots),
      trapped:value.trapped, maybe_trapped:value.maybeTrapped,
      maybe_disabled:value.maybeDisabled, maybe_locked:value.maybeLocked,
      switch_flag:value.switchFlag, force_switch_flag:value.forceSwitchFlag,
      last_move:value.lastMove?.id || null, last_move_used:value.lastMoveUsed?.id || null,
      move_this_turn:value.moveThisTurn, hp_type:value.hpType, hp_power:value.hpPower,
      base_hp_type:value.baseHpType, base_hp_power:value.baseHpPower};
  }
  return {ok:true, audit_schema:1, version, generation:battle.gen,
    turn:battle.turn, ended:battle.ended, winner:battle.winner || null,
    rng_seed:copy(battle.prng.rng.seed), starting_seed:battle.prng.startingSeed,
    log_offset:logOffset, protocol:battle.log.slice(),
    field:{weather:battle.field.weather, weather_state:copy(battle.field.weatherState)},
    request_state:battle.requestState,
    sides:battle.sides.map((side, index) => ({id:side.id, name:side.name, index,
      active_slot:slots.get(side.active[0])?.slot || null,
      order:side.pokemon.map(value => slots.get(value).slot),
      request:copy(side.activeRequest), choice:copy(side.choice),
      side_conditions:copy(side.sideConditions),
      party:original[index].map(pokemon)}))};
}

function handle(request) {
  return request.command === 'audit_inspect' ? inspect() : production.handle(request);
}

if (require.main === module) {
  const input = readline.createInterface({input:process.stdin, crlfDelay:Infinity});
  input.on('line', line => {
    try {process.stdout.write(JSON.stringify(handle(JSON.parse(line))) + '\n');}
    catch (error) {process.stdout.write(JSON.stringify({ok:false, error:error.message, stack:error.stack}) + '\n');}
  });
  input.on('close', () => {
    const {battle} = production.auditSource();
    if (battle) battle.destroy();
  });
}

module.exports = {handle, inspect};
