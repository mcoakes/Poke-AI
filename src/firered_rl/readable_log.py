"""Readable names and values from the actual encoded policy observation."""
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

from firered_rl.turn_env import POCKETS


@lru_cache(maxsize=8)
def names(profile):
    if profile != "firered_us_v1_0":
        return {}
    path = Path(__file__).resolve().parents[2] / "mgba/profiles/firered_us_v1_0_names.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def status(raw):
    labels = (["Sleep"] if raw & 7 else []) + [
        label for bit, label in [(8, "Poison"), (16, "Burn"), (32, "Freeze"),
                                (64, "Paralysis"), (128, "Bad poison")] if raw & bit]
    return "/".join(labels) or "Healthy"


def describe_observation(obs, profile):
    table = names(profile)
    def name(group, value):
        return table.get(group, {}).get(str(int(value)), f"#{int(value)}")
    def moves(rows):
        return "; ".join(f"{i+1}: {name('moves', row[0])} [{name('types', row[1])}] "
                         f"PP {row[2]}/{row[3]}" for i, row in enumerate(rows) if row[0]) or "None"
    def action(value):
        if value is None:
            return "Human controls"
        value = int(value)
        if value == 10:
            return "WAIT (automatic action/replacement interval)"
        if value >= 4:
            slot = value-4
            return f"Switch to party slot {slot+1}: {name('species', obs['party'][slot][1])}"
        slot = int(obs["active"][0][8])-1
        move = obs["moves"][slot][value][0] if 0 <= slot < 6 else 0
        return f"Move slot {value+1}: {name('moves', move)}"
    turn, forced, terminal, outcome = obs["context"]
    lines = [f"Turn {turn} | switch required={bool(forced)} | terminal={bool(terminal)} | outcome code={outcome}"]
    for side, row in zip(("Player", "Opponent"), obs["active"]):
        if not row[0]:
            lines.append(f"  {side} active: unavailable/absent")
            continue
        hp = f"{row[3]}/{row[4]}" if side == "Player" else f"{row[3]}%"
        slot = row[8] if side == "Player" else obs["opponent_active_index"][0]
        lines.append(f"  {side} active: {name('species', row[1])} Lv{row[2]} | slot {slot} | HP {hp} | "
                     f"{name('types', row[5])}/{name('types', row[6])} | {status(row[7])} (0x{row[7]:08x})")
        if side == "Player":
            lines.append(f"    Volatile status: 0x{row[9]:08x}")
    lines.append("  Player party (stored HP; active HP above may differ during transitions):")
    for i, row in enumerate(obs["party"]):
        if not row[0]:
            lines.append(f"    {i+1}: Empty")
            continue
        lines.append(f"    {i+1}: {name('species', row[1])} Lv{row[2]} HP {row[3]}/{row[4]} | "
                     f"{name('types', row[5])}/{name('types', row[6])} | {status(row[7])} "
                     f"(0x{row[7]:08x}) | Held: {name('items', row[8])} | Egg={bool(row[9])}")
        lines.append("      " + moves(obs["moves"][i]))
    lines.append("  Opponent party (species/moves exposed to policy; bench HP is not in observation):")
    for i, row in enumerate(obs["opponent_party"]):
        lines.append(f"    {i+1}: {name('species', row[1]) if row[0] else 'Empty'}")
        if row[0]:
            lines.append("      " + moves(obs["opponent_moves"][i]))
    for pocket, rows in zip(POCKETS, obs["bag"]):
        entries = "; ".join(f"slot {i+1}: {name('items', row[0])} x{row[1]}"
                            for i, row in enumerate(rows) if row[0])
        lines.append(f"  Bag {pocket}: {entries or 'Empty'}")
    lines.append("  Legal actions: " + "; ".join(f"{i}={action(i)}" for i, valid in enumerate(obs["action_mask"]) if valid))
    return lines, action


def format_event(event, episode, reports, observations, fields):
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"\n[{stamp}] {episode} | {event.upper()} | decision {fields.get('decision', 0)}"]
    if event == "failed":
        return "\n".join(lines + [f"ERROR: {fields['error']}"]) + "\n"
    for i, (report, obs) in enumerate(zip(reports, observations)):
        lines.append(f"PLAYER {i+1} | frame {report.get('frame', '?')} | "
                     f"menu {next(iter(report['battle'].get('menus', [])), {}).get('state', '?')}")
        details, action = describe_observation(obs, report.get("profile"))
        lines.extend(details)
        if event == "submitted":
            selected = fields["actions"][i]
            lines.append(f"  CHOSEN ACTION: {selected} = {action(selected)} (submitted; result pending)")
        if event == "completed":
            lines.append(f"  REWARD: {fields['rewards'][i]:+g} raw points | "
                         f"components={fields.get('reward_components', [None, None])[i]}")
    if event == "completed":
        lines.append(f"RESULT: terminated={fields['terminated']} truncated={fields['truncated']}")
    return "\n".join(lines) + "\n"
