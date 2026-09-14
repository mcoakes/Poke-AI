"use strict";

const $ = (id) => document.getElementById(id);
const clone = (value) => JSON.parse(JSON.stringify(value));
const sides = [
  {label: "Your party", mode: "saved", category: "competitive", size: 3, party: [], origin: "", savedCategory: null, savedTeam: "", teamSearch: ""},
  {label: "The challenger", mode: "saved", category: "competitive", size: 3, party: [], origin: "", savedCategory: null, savedTeam: "", teamSearch: ""},
];
const labels = {competitive: "Competitive", casual: "Casual", random: "Completely random", mixed: "Mixed categories"};
const statNames = {hp: "HP", atk: "Attack", def: "Defense", spa: "Sp. Atk", spd: "Sp. Def", spe: "Speed"};
let catalog, settings, token, speciesByName, editing, currentJob, pollTimer;

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}
function button(text, className, action) {
  const item = node("button", className, text);
  item.type = "button";
  item.addEventListener("click", action);
  return item;
}
function option(value, label) {
  const item = node("option", "", label ?? value);
  item.value = value;
  return item;
}
function showError(id, message) { $(id).textContent = message; $(id).hidden = !message; }
function picture(mon, sizeClass = "") {
  const img = node("img", sizeClass);
  img.src = speciesByName[mon.species || mon.name].sprite;
  img.alt = mon.species || mon.name;
  img.loading = "lazy";
  img.width = 96; img.height = 96;
  return img;
}
function types(names) {
  const list = node("div", "type-list");
  names.forEach((type) => list.append(node("span", `type-chip type-${type.toLowerCase()}`, type)));
  return list;
}
async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Play-Token": token}, body: JSON.stringify(payload),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}

function savedControls(side, index) {
  const wrap = node("div", "saved-controls");
  const categories = catalog.team_categories;
  const refresh = button("↻ Refresh teams", "quiet", async () => {
    refresh.disabled = true;
    try {
      const result = await api("/api/teams");
      catalog.team_categories = result.categories;
      drawTeams();
    } catch (error) { showError(`team-error-${index}`, error.message); refresh.disabled = false; }
  });
  refresh.setAttribute("aria-label", `Refresh teams for player ${index + 1}`);
  if (!categories.length) {
    wrap.append(node("p", "mode-hint", "No saved teams found. Add a team JSON under config, then refresh."), refresh);
    return wrap;
  }
  if (!categories.some(group => group.id === side.savedCategory)) side.savedCategory = categories[0].id;
  const group = categories.find(group => group.id === side.savedCategory);
  const filters = node("div", "saved-filters");
  const categoryLabel = node("label", "", "Category");
  const category = node("select"); category.setAttribute("aria-label", `Saved team category for player ${index + 1}`);
  categories.forEach(value => category.append(option(value.id, `${value.label} (${value.teams.length})`)));
  category.value = side.savedCategory;
  category.onchange = () => { side.savedCategory = category.value; side.savedTeam = ""; side.teamSearch = ""; drawTeams(); };
  categoryLabel.append(category);
  const searchLabel = node("label", "", "Search this category");
  const search = node("input"); search.type = "search"; search.placeholder = "Team or Pokémon name";
  search.setAttribute("aria-label", `Search saved teams for player ${index + 1}`);
  search.value = side.teamSearch;
  searchLabel.append(search); filters.append(categoryLabel, searchLabel);
  const choice = node("div", "saved-choice"), teamLabel = node("label", "", "Team");
  const teams = node("select"); teams.setAttribute("aria-label", `Saved team for player ${index + 1}`);
  teams.onchange = () => { side.savedTeam = teams.value; };
  teamLabel.append(teams);
  const use = button("Use team", "secondary", async () => {
    use.disabled = true; use.textContent = "Loading…";
    try {
      const result = await api("/api/team", {id: side.savedTeam});
      side.party = result.team.party;
      side.origin = `${result.team.name} · ${result.category} / ${result.template}`;
      drawTeams();
    } catch (error) { showError(`team-error-${index}`, error.message); use.disabled = false; use.textContent = "Use team"; }
  });
  use.setAttribute("aria-label", `Use saved team for player ${index + 1}`);
  function filterTeams() {
    const query = side.teamSearch.trim().toLowerCase();
    const matches = group.teams.filter(team => `${team.name} ${team.file} ${team.species.join(" ")}`.toLowerCase().includes(query));
    teams.replaceChildren(...matches.map(team => option(team.id, `${team.name} · ${team.size} Pokémon · ${team.file}`)));
    if (matches.some(team => team.id === side.savedTeam)) teams.value = side.savedTeam;
    if (!matches.length) teams.append(option("", "No teams match your search"));
    side.savedTeam = teams.value;
    teams.disabled = !matches.length; use.disabled = !matches.length;
  }
  search.oninput = () => { side.teamSearch = search.value; filterTeams(); };
  filterTeams();
  choice.append(teamLabel, use);
  const foot = node("div", "saved-footer");
  foot.append(node("p", "hint", "Use a team to preview it, then edit any slot."), refresh);
  wrap.append(filters, choice, foot);
  if (side.origin) wrap.append(node("p", "team-origin", `Loaded: ${side.origin}`));
  return wrap;
}

function drawTeams() {
  $("teams").replaceChildren();
  sides.forEach((side, index) => {
    const panel = node("section", `team-panel ${index ? "enemy" : ""}`);
    panel.setAttribute("aria-label", side.label);
    const heading = node("div", "team-heading");
    const title = node("div");
    title.append(node("p", "eyebrow", index ? "THE SAVED POLICY" : "YOU’RE IN CONTROL"), node("h2", "", side.label));
    heading.append(title, node("span", "player-pill", `PLAYER ${index + 1}`));
    const controls = node("div", "team-controls");
    const modes = node("div", "segmented");
    [["saved", "Saved teams"], ["custom", "Build a team"], ["random", "Random team"]].forEach(([value, text]) => {
      const b = button(text, side.mode === value ? "selected" : "", () => { side.mode = value; drawTeams(); });
      b.setAttribute("aria-pressed", String(side.mode === value));
      b.setAttribute("aria-label", `${text} for player ${index + 1}`);
      modes.append(b);
    });
    controls.append(modes);
    if (side.mode === "saved") {
      controls.append(savedControls(side, index));
    } else if (side.mode === "random") {
      const randomOptions = node("div", "random-options");
      const categoryLabel = node("label", "", "Category");
      const category = node("select"); category.setAttribute("aria-label", `Category for player ${index + 1}`);
      Object.entries(labels).forEach(([value, name]) => category.append(option(value, name)));
      category.value = side.category;
      category.onchange = () => { side.category = category.value; side.origin = "Settings changed — roll a team to apply."; drawTeams(); };
      categoryLabel.append(category);
      const sizeLabel = node("label", "", "Party size");
      const size = node("select"); size.setAttribute("aria-label", `Party size for player ${index + 1}`);
      for (let i = 1; i <= 6; i++) size.append(option(String(i)));
      size.value = side.size;
      size.onchange = () => { side.size = Number(size.value); side.origin = "Settings changed — roll a team to apply."; drawTeams(); };
      sizeLabel.append(size);
      const roll = button("↻ Roll team", "secondary", async () => {
        roll.disabled = true; roll.textContent = "Rolling…";
        try {
          const result = await api("/api/team", {category: side.category, size: side.size});
          side.party = result.team.party;
          side.origin = `${labels[result.category]} · ${result.template}`;
          drawTeams();
        } catch (error) { showError(`team-error-${index}`, error.message); roll.disabled = false; roll.textContent = "↻ Roll team"; }
      });
      roll.setAttribute("aria-label", `Roll team for player ${index + 1}`);
      randomOptions.append(categoryLabel, sizeLabel, roll);
      controls.append(randomOptions);
      controls.append(node("p", "team-origin", side.origin || "Roll to preview a team. Click any Pokémon to edit it."));
    } else {
      controls.append(node("p", "mode-hint", "Choose a Pokémon for each slot. Click a party member to fine-tune their moves and stats."));
    }
    const grid = node("div", "party-grid");
    for (let slot = 0; slot < 6; slot++) {
      const mon = side.party[slot];
      const card = button("", `slot ${mon ? "" : "empty"}`, () => openSlot(index, slot));
      card.setAttribute("aria-label", mon ? `Edit ${mon.species}, player ${index + 1}, slot ${slot + 1}` : `Add Pokémon, player ${index + 1}, slot ${slot + 1}`);
      card.append(node("span", "slot-number", slot === 0 ? "01 · LEAD" : String(slot + 1).padStart(2, "0")));
      if (mon) {
        const head = node("div", "slot-head");
        const description = node("div");
        description.append(node("div", "slot-name", mon.species), node("div", "slot-level", `Lv. ${mon.level} · ${mon.held_item || "None"}`), types(speciesByName[mon.species].types));
        head.append(picture(mon), description);
        card.append(head, node("p", "slot-moves", mon.moves.join(" · ")));
      } else {
        card.append(node("span", "plus", "+"), node("span", "empty-label", "Add Pokémon"));
      }
      grid.append(card);
    }
    const error = node("p", "notice error team-error"); error.id = `team-error-${index}`; error.hidden = true;
    panel.append(heading, controls, grid, error); $("teams").append(panel);
  });
  const ready = sides.every((side) => side.party.length > 0);
  $("match-summary").textContent = ready ? `${sides[0].party.length} vs ${sides[1].party.length} Pokémon · Your team leads with ${sides[0].party[0].species}` : "Add at least one Pokémon to each party";
  $("review").disabled = !ready || !$("checkpoint").value;
}

function defaultMon(name) {
  const species = speciesByName[name];
  const preset = catalog.presets[name];
  const moves = preset ? preset.moves.slice() : [...species.moves].sort((a, b) => {
    const score = (move) => Math.min(catalog.moves[move].power, 100) + (species.types.includes(catalog.moves[move].type) ? 40 : 0);
    return score(b) - score(a);
  }).slice(0, 4);
  // Sparse natural move pools can still be edited with the experimental all-moves list.
  for (const move of ["Tackle", "Protect", "Toxic", "Rest"]) {
    if (moves.length < 4 && !moves.includes(move)) moves.push(move);
  }
  return {species: name, level: 50, moves, nature: "Hardy", held_item: "None", evs: Object.fromEntries(catalog.stats.map(s => [s, 0])), ivs: Object.fromEntries(catalog.stats.map(s => [s, 31]))};
}

function openSlot(side, slot) {
  editing = {side, slot: Math.min(slot, sides[side].party.length), draft: clone(sides[side].party[slot] || null), lead: false};
  if (editing.draft) openEditor();
  else openPicker();
}
function openPicker() {
  $("species-search").value = ""; $("type-filter").value = "";
  drawSpecies(); $("picker").showModal(); $("species-search").focus();
}
function drawSpecies() {
  const text = $("species-search").value.trim().toLowerCase();
  const type = $("type-filter").value;
  const filtered = catalog.species.filter(s => (!text || s.name.toLowerCase().includes(text) || String(s.dex).padStart(3, "0").includes(text.replace("#", ""))) && (!type || s.types.includes(type)));
  $("species-count").textContent = `${filtered.length} Pokémon · Generations I–III`;
  $("species-grid").replaceChildren();
  filtered.forEach(species => {
    const card = button("", "species-card", () => { editing.draft = defaultMon(species.name); $("picker").close(); openEditor(); });
    card.setAttribute("aria-label", `Choose ${species.name}`);
    card.append(picture(species), node("strong", "", species.name), node("small", "", `#${String(species.dex).padStart(3, "0")} · ${species.types.join(" / ")}`));
    $("species-grid").append(card);
  });
}

function openEditor() {
  const mon = editing.draft;
  $("editor-title").textContent = mon.species;
  $("editor-side").textContent = editing.side ? "THE AI’S PARTY" : "YOUR PARTY";
  $("editor-sprite").src = speciesByName[mon.species].sprite;
  $("editor-sprite").alt = mon.species;
  $("editor-types").replaceChildren(types(speciesByName[mon.species].types));
  $("level").value = mon.level;
  $("nature").value = mon.nature || "Hardy";
  $("held-item").value = mon.held_item || "None";
  $("all-moves").checked = mon.moves.some(move => !speciesByName[mon.species].moves.includes(move));
  $("moves").replaceChildren();
  for (let i = 0; i < 4; i++) {
    const row = node("div", "move-field"), label = node("label", "", `Move ${i + 1}`);
    const input = node("input"); input.id = `move-${i}`; input.value = mon.moves[i] || "";
    input.setAttribute("list", "move-list"); input.autocomplete = "off"; input.required = i === 0;
    const detail = node("div", "move-type"); detail.id = `move-detail-${i}`;
    input.addEventListener("input", () => describeMove(i));
    label.append(input, detail); row.append(node("span", "move-num", String(i + 1).padStart(2, "0")), label); $("moves").append(row);
  }
  updateMoveOptions();
  for (let i = 0; i < 4; i++) describeMove(i);
  $("stats").replaceChildren();
  catalog.stats.forEach(stat => {
    const column = node("div", "stat-cell"); column.append(node("p", "stat-name", statNames[stat]));
    for (const kind of ["ivs", "evs"]) {
      const label = node("label", "", kind === "ivs" ? "IV" : "EV");
      const input = node("input"); input.type = "number"; input.min = "0"; input.max = kind === "ivs" ? "31" : "255";
      input.id = `${kind}-${stat}`; input.value = mon[kind]?.[stat] ?? (kind === "ivs" ? 31 : 0); input.required = true;
      input.setAttribute("aria-label", `${statNames[stat]} ${kind === "ivs" ? "IV" : "EV"}`);
      input.addEventListener("input", updateEVs); label.append(input); column.append(label);
    }
    $("stats").append(column);
  });
  $("preset").disabled = !catalog.presets[mon.species];
  const suggestion = catalog.presets[mon.species];
  $("preset").textContent = suggestion?.source === "experimental" ? "Apply experimental set" : "Apply suggested set";
  $("preset").title = suggestion?.note || "Apply a curated Gen III moveset, nature, EVs and held item";
  $("remove-mon").hidden = !sides[editing.side].party[editing.slot];
  $("make-lead").hidden = editing.slot === 0;
  $("make-lead").textContent = editing.lead ? "Lead selected ✓" : "Make lead";
  updateEVs(); showError("editor-error", "");
  if (!$("editor").open) $("editor").showModal();
}
function canonical(name, choices) {
  return choices.find(key => key.toLowerCase() === name.trim().toLowerCase());
}
function describeMove(index) {
  const name = canonical($(`move-${index}`).value, Object.keys(catalog.moves));
  const info = catalog.moves[name];
  $(`move-detail-${index}`).textContent = info ? `${info.type} · ${info.category}${info.power ? ` · ${info.power} power` : ""}` : "Select a Gen III move";
}
function updateMoveOptions() {
  const pool = $("all-moves").checked ? Object.keys(catalog.moves) : speciesByName[editing.draft.species].moves;
  $("move-list").replaceChildren(...[...pool].sort().map(name => option(name, `${name} · ${catalog.moves[name].type}`)));
}
function updateEVs() {
  const total = catalog.stats.reduce((sum, stat) => sum + Number($(`evs-${stat}`).value || 0), 0);
  $("ev-total").textContent = `EV total: ${total} / 510`;
  $("ev-total").classList.toggle("invalid", total > 510);
}
function readEditor() {
  const mon = clone(editing.draft);
  mon.level = Number($("level").value); mon.nature = $("nature").value;
  mon.held_item = canonical($("held-item").value, Object.keys(catalog.items));
  const moveNames = Array.from({length: 4}, (_, i) => $(`move-${i}`).value.trim());
  while (moveNames.length && !moveNames.at(-1)) moveNames.pop();
  mon.moves = moveNames.map(name => canonical(name, Object.keys(catalog.moves)));
  if (!Number.isInteger(mon.level) || mon.level < 1 || mon.level > 100) throw new Error("Level must be between 1 and 100.");
  if (!mon.held_item) throw new Error("Choose a known held item, or None.");
  if (!mon.moves.length || mon.moves.some(move => !move)) throw new Error("Choose one to four Gen III moves, starting in slot 1 without gaps.");
  if (new Set(mon.moves).size !== mon.moves.length) throw new Error("Each move must be different.");
  for (const kind of ["ivs", "evs"]) {
    mon[kind] = Object.fromEntries(catalog.stats.map(stat => [stat, Number($(`${kind}-${stat}`).value)]));
    if (Object.values(mon[kind]).some(value => !Number.isInteger(value) || value < 0 || value > (kind === "ivs" ? 31 : 255))) throw new Error(`${kind.toUpperCase()}: enter values within the displayed limits.`);
  }
  if (Object.values(mon.evs).reduce((a, b) => a + b, 0) > 510) throw new Error("EVs cannot total more than 510. Reduce a stat or use Clear EVs.");
  return mon;
}

function teamMini(side, heading) {
  const wrap = node("div"); wrap.append(node("h3", "", heading));
  side.party.forEach((mon, i) => {
    const line = node("div", "mini-mon"); line.append(picture(mon), node("span", "", `${mon.species} · Lv. ${mon.level}${i ? "" : " · lead"}`)); wrap.append(line);
  });
  return wrap;
}
function review() {
  const playerName = $("player-name").value.trim();
  if (playerName && !/^[A-Za-z0-9 ]{1,7}$/.test(playerName)) {
    showError("connection-error", "Your trainer name must use 1–7 letters, digits, or spaces.");
    $("player-name").focus(); return;
  }
  const speed = $("speed").value.trim();
  if (speed !== "unlimited" && (!Number.isFinite(Number(speed)) || Number(speed) <= 0)) {
    $("settings").hidden = false; $("settings-toggle").setAttribute("aria-expanded", "true"); $("speed").focus();
    showError("connection-error", "Battle speed must be a positive number or unlimited."); return;
  }
  showError("connection-error", ""); showError("confirm-error", "");
  $("confirm-teams").replaceChildren(teamMini(sides[0], playerName ? `You · ${playerName}` : "You · Player 1"), teamMini(sides[1], "AI · Player 2"));
  $("confirm-checkpoint").textContent = `Checkpoint: ${$("checkpoint").selectedOptions[0].textContent}`;
  $("confirm-workers").textContent = `2 team saves · ${$("workers").value} worker(s) · ${speed === "unlimited" ? "Unlimited" : `${speed}×`} battle speed`;
  $("confirm").showModal();
}

function showPreview() {
  $("match-preview").replaceChildren();
  sides.forEach((side, i) => {
    if (i) $("match-preview").append(node("span", "versus", "VS"));
    const block = node("div", "preview-side"), sprites = node("div", "preview-sprites");
    side.party.forEach(mon => sprites.append(picture(mon)));
    block.append(node("p", "hint", i ? "THE AI" : "YOUR PARTY"), sprites); $("match-preview").append(block);
  });
}
function drawJob(job) {
  if (!job) return;
  if ((!currentJob || currentJob.id !== job.id) && job.teams) {
    job.teams.forEach((team, index) => { sides[index].party = clone(team.party); });
    showPreview();
  }
  currentJob = job;
  $("builder").hidden = true; $("progress").hidden = false;
  const finished = ["complete", "failed", "cancelled"].includes(job.status);
  const playing = job.status === "playing", connecting = job.status === "connecting";
  $("progress-eyebrow").textContent = finished ? "MATCH REPORT" : playing ? "BATTLE READY" : "MATCH IN PREPARATION";
  $("progress-title").textContent = playing ? "Your move." : job.status === "complete" ? "Good game." : job.status === "failed" ? "Let’s check the setup." : job.status === "cancelled" ? "Match stopped." : "Your next battle is on its way.";
  $("progress-message").textContent = job.message;
  $("save-count").textContent = `${job.states_ready} of 2 states verified · ${job.workers} worker(s)`;
  $("stage-saves").className = job.states_ready === 2 ? "done" : finished ? "" : "active";
  $("stage-link").className = playing || job.status === "complete" ? "done" : connecting ? "active" : "";
  $("stage-battle").className = job.status === "complete" ? "done" : playing ? "active" : "";
  document.querySelectorAll(".stepper li").forEach(li => li.classList.toggle("current", li.dataset.stage === (playing || job.status === "complete" ? "play" : "prepare")));
  $("cancel-job").hidden = finished; $("cancel-job").disabled = job.status === "stopping";
  $("new-match").hidden = !finished;
  showError("job-error", job.error || "");
  $("job-directory").textContent = `Match files: ${job.directory}`;
  $("generation-log").textContent = job.generation_log || "Waiting for generation…";
  $("battle-log").textContent = job.battle_log || "Waiting for verified states…";
  $("result").hidden = !job.result || job.status !== "complete";
  if (job.result && job.status === "complete") {
    const result = job.result;
    $("result").replaceChildren(node("span", "", result.truncated ? "Battle decision limit reached" : `You ${result.human_outcome}.`), node("small", "", `${result.decisions} decisions · Human ${result.human_reward ?? 0} · AI ${result.ai_reward ?? 0}`));
  }
}
async function poll() {
  clearTimeout(pollTimer);
  try {
    const {job} = await api("/api/job");
    showError("connection-error", "");
    if (job) drawJob(job);
    if (!job || ["complete", "failed", "cancelled"].includes(job.status)) return;
  } catch (error) {
    showError("connection-error", "The local server is unreachable. Check its terminal; the browser will keep trying.");
  }
  pollTimer = setTimeout(poll, 750);
}

async function load() {
  try {
    const data = await api("/api/catalog");
    catalog = data.catalog; settings = data.settings; token = data.token;
    speciesByName = Object.fromEntries(catalog.species.map(s => [s.name, s]));
    $("checkpoint").replaceChildren(...settings.checkpoints.map(c => option(c.id, c.label)));
    if (!settings.checkpoints.length) $("checkpoint").append(option("", "No checkpoint available"));
    $("no-model").hidden = !!settings.checkpoints.length;
    $("speed").value = settings.speed;
    for (let i = 1; i <= settings.workers; i++) $("workers").append(option(String(i), `${i} worker${i === 1 ? "" : "s"}`));
    $("workers").value = settings.workers;
    $("capture-hint").textContent = `Capture: ${settings.capture_speed} · Link setup: ${settings.setup_speed}. Battle speed is separate.`;
    catalog.natures.forEach(n => $("nature").append(option(n)));
    Object.keys(catalog.items).sort().forEach(n => $("item-list").append(option(n)));
    [...new Set(catalog.species.flatMap(s => s.types))].sort().forEach(t => $("type-filter").append(option(t)));
    $("loading").hidden = true; $("builder").hidden = false; drawTeams();
    await poll();
  } catch (error) {
    $("loading").hidden = true; showError("connection-error", `Could not load Battle Studio: ${error.message}. Check the launcher terminal, then refresh.`);
  }
}

document.querySelectorAll("[data-close]").forEach(b => b.onclick = () => $(b.dataset.close).close());
$("settings-toggle").onclick = () => { $("settings").hidden = !$("settings").hidden; $("settings-toggle").setAttribute("aria-expanded", String(!$("settings").hidden)); };
$("species-search").addEventListener("input", drawSpecies);
$("type-filter").addEventListener("change", drawSpecies);
$("all-moves").addEventListener("change", updateMoveOptions);
$("max-ivs").onclick = () => catalog.stats.forEach(s => { $(`ivs-${s}`).value = 31; });
$("clear-evs").onclick = () => { catalog.stats.forEach(s => { $(`evs-${s}`).value = 0; }); updateEVs(); };
$("make-lead").onclick = () => { editing.lead = !editing.lead; $("make-lead").textContent = editing.lead ? "Lead selected ✓" : "Make lead"; };
$("change-species").onclick = () => { $("editor").close(); openPicker(); };
$("preset").onclick = () => {
  const set = catalog.presets[editing.draft.species];
  const ivs = Object.fromEntries(catalog.stats.map(s => [s, Number($(`ivs-${s}`).value)]));
  editing.draft = {...editing.draft, level: Number($("level").value), ivs, moves: clone(set.moves), nature: set.nature, evs: clone(set.evs), held_item: set.items[0]};
  openEditor();
};
$("remove-mon").onclick = () => { sides[editing.side].party.splice(editing.slot, 1); sides[editing.side].mode = "custom"; sides[editing.side].origin = ""; $("editor").close(); drawTeams(); };
$("editor-form").onsubmit = (event) => {
  event.preventDefault();
  try {
    const mon = readEditor(), side = sides[editing.side];
    side.party[editing.slot] = mon;
    if (editing.lead) { side.party.splice(editing.slot, 1); side.party.unshift(mon); }
    side.mode = "custom"; side.origin = "";
    $("editor").close(); drawTeams();
  } catch (error) { showError("editor-error", error.message); }
};
$("checkpoint").onchange = drawTeams;
$("refresh-models").onclick = async () => {
  const previous = $("checkpoint").value;
  try {
    const data = await api("/api/catalog"); token = data.token;
    $("checkpoint").replaceChildren(...data.settings.checkpoints.map(c => option(c.id, c.label)));
    if (data.settings.checkpoints.some(c => c.id === previous)) $("checkpoint").value = previous;
    $("no-model").hidden = !!data.settings.checkpoints.length; drawTeams();
  } catch (error) { showError("connection-error", error.message); }
};
$("review").onclick = review;
$("start-job").onclick = async () => {
  $("start-job").disabled = true; showError("confirm-error", "");
  try {
    const {job} = await api("/api/jobs", {player_name: $("player-name").value.trim(), teams: sides.map((s, i) => ({name: i ? "AI team" : "Your team", party: s.party})), checkpoint: $("checkpoint").value, workers: Number($("workers").value), speed: $("speed").value.trim(), stochastic: $("stochastic").checked});
    $("confirm").close(); showPreview(); drawJob(job); poll();
  } catch (error) { showError("confirm-error", error.message); }
  finally { $("start-job").disabled = false; }
};
$("cancel-job").onclick = async () => {
  $("cancel-job").disabled = true;
  try { const {job} = await api("/api/job/cancel", {}); drawJob(job); poll(); }
  catch (error) { showError("job-error", error.message); $("cancel-job").disabled = false; }
};
$("new-match").onclick = () => {
  clearTimeout(pollTimer); currentJob = null;
  $("progress").hidden = true; $("builder").hidden = false; drawTeams();
  document.querySelectorAll(".stepper li").forEach(li => li.classList.toggle("current", li.dataset.stage === "build"));
};

load();
