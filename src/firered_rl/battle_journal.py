"""A compact spectator transcript, based on actual game messages and HP events."""
from datetime import datetime, timezone

from firered_rl.readable_log import names


class BattleJournal:
    def __init__(self, path):
        self.path = path
        self.seen = 0
        self.hp_line = None
        self.error = None

    def write(self, lines):
        if lines:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as log:
                log.write("\n".join(lines) + "\n")

    def species(self, value):
        return self.names.get("species", {}).get(str(value), f"Pokemon #{value}")

    def hp(self, mons):
        entries = {self.players[m["battler_id"]]:
                   f"P{self.players[m['battler_id']]} {self.species(m['species_id'])} {m['current_hp']}/{m['max_hp']}"
                   for m in mons if m["battler_id"] in self.players}
        if len(entries) != 2:
            return []
        line = "  HP  |  " + "  |  ".join(entries[i] for i in (1, 2))
        if line == self.hp_line:
            return []
        self.hp_line = line
        return [line]

    def start(self, episode, reports):
        self.seen, self.hp_line, self.error = 0, None, None
        self.names = names(reports[0].get("profile"))
        self.players = {next(m for m in r["battle"]["battlers"] if m["side"] == "player")["battler_id"]: i+1
                        for i, r in enumerate(reports)}
        self.master = next(i for i, r in enumerate(reports) if r["battle"]["type_flags"] & 4)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.write(["", f"=== {episode} | {stamp} ===",
                    f"Game text: Player {self.master+1}'s view; foe/enemy = Player {2-self.master}."] +
                   self.hp(reports[self.master]["battle"]["battlers"]))

    def collect(self, states):
        combat = states[self.master].get("combat") or {}
        lines = []
        if combat.get("error") and combat["error"] != self.error:
            self.error = combat["error"]
            lines.append(f"[Battle event capture stopped: {self.error}]")
        for event in combat.get("events", []):
            seq = event["seq"]
            if seq <= self.seen:
                continue
            if seq > self.seen+1:
                lines.append(f"[{seq-self.seen-1} battle events were unavailable]")
            self.seen = seq
            if event["kind"] == "text":
                lines.append("  " + event["text"])
            elif event["kind"] == "hp":
                player = self.players.get(event["battler_id"], "?")
                change = event["after"]-event["before"]
                effect = f"took {-change} damage" if change < 0 else f"recovered {change} HP"
                lines.append(f"  P{player} {self.species(event['species_id'])} {effect}. "
                             f"({event['before']} → {event['after']}/{event['max_hp']} HP)")
            lines.extend(self.hp(event.get("hp", [])))
        self.write(lines)

    def record(self, event, episode, reports, fields):
        if event == "reset":
            self.start(episode, reports)
        elif event == "submitted":
            self.write(["", f"Turn {reports[self.master]['battle']['turn']+1} · decision {fields['decision']}"])
        elif event == "completed":
            lines = self.hp(reports[self.master]["battle"]["battlers"])
            if fields.get("terminated"):
                lines.append("RESULT: " + " | ".join(f"P{i+1} {r['battle']['outcome']}" for i, r in enumerate(reports)))
            elif fields.get("truncated"):
                lines.append("Battle stopped at the decision/time limit.")
            self.write(lines)
        elif event == "failed":
            self.write([f"Battle interrupted: {fields['error']}"])
