"""Local team-builder service. Games still run in the visible mGBA frontend."""
from __future__ import annotations

import copy
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import random
import secrets
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit

from .launch_options import speed_value
from .checkpoints import is_latest, resolve_checkpoint
from .teams import NATURES, PokemonSpec, STATS

CATEGORIES = ("competitive", "casual", "random")
FINISHED = {"complete", "failed", "cancelled"}


def positive_int(value, label, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"{label} must be a positive integer" + (f" up to {maximum}" if maximum else ""))
    return value


class TeamCatalog:
    def __init__(self, project):
        self.project = Path(project)
        self.data = json.loads((self.project / "config/generation/firered_us_v1_0.json").read_text())
        presets = json.loads((self.project / "config/generation/competitive_gen3.json").read_text())["sets"]
        fallback = json.loads((self.project / "config/generation/suggested_gen3.json").read_text())["sets"]
        self.presets = {p["species"]: p for p in fallback}
        self.presets.update({p["species"]: dict(p, source="curated") for p in presets})
        self.names = {kind: {self.name_key(name): name for name in self.data[kind]}
                      for kind in ("species", "moves", "items")}
        self.team_files = {}

    @staticmethod
    def name_key(name):
        return "".join(name.casefold().split())

    def canonical(self, name, kind):
        return self.names[kind].get(self.name_key(name), name)

    def public(self):
        species = []
        for name, info in self.data["species"].items():
            species.append(dict(name=name, dex=info["id"], types=info["types"],
                                moves=info["moves"], sprite=f"/sprites/{info['id']}.png"))
        return dict(species=sorted(species, key=lambda p: p["dex"]), moves=self.data["moves"],
                    items=self.data["items"], natures=NATURES, presets=self.presets,
                    categories=[*CATEGORIES, "mixed"], stats=STATS,
                    team_categories=self.available_teams())

    def validate(self, team):
        if not isinstance(team, dict) or set(team) - {"name", "party"}:
            raise ValueError("a team must contain a name and party")
        party = team.get("party")
        if not isinstance(party, list) or not 1 <= len(party) <= 6:
            raise ValueError("each team needs one to six Pokémon")
        name = team.get("name", "Custom team")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("team name must contain 1–80 characters")
        result = []
        for value in party:
            mon = PokemonSpec.from_dict(value)
            species = self.canonical(mon.species, "species")
            moves = [self.canonical(move, "moves") for move in mon.moves]
            item = self.canonical(mon.held_item, "items")
            if species not in self.data["species"]:
                raise ValueError(f"unknown Gen III species: {mon.species}")
            if any(move not in self.data["moves"] for move in moves):
                raise ValueError(f"{mon.species}: unknown Gen III move")
            if len(set(moves)) != len(moves):
                raise ValueError(f"{mon.species}: duplicate moves")
            if item not in self.data["items"]:
                raise ValueError(f"{mon.species}: unknown Gen III held item")
            result.append(dict(species=species, level=mon.level, moves=moves,
                               evs=dict(zip(STATS, mon.evs)), ivs=dict(zip(STATS, mon.ivs)),
                               held_item=item, nature=mon.nature or "Hardy"))
        return dict(name=name.strip(), party=result)

    def read_team(self, path):
        if not path.resolve().is_relative_to((self.project / "config").resolve()):
            raise ValueError("team must be inside config")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or "party" not in value:
            raise ValueError("not a team JSON")
        # Generator metadata is not part of the editable party specification.
        return self.validate({"name": value.get("name", path.stem), "party": value["party"]})

    def available_teams(self):
        """Index real team files throughout config without sending every moveset."""
        root = self.project / "config"
        groups, files = {}, {}
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() != ".json" or not path.is_file():
                continue
            try:
                team = self.read_team(path)
            except (ValueError, TypeError, OSError, RuntimeError):
                continue  # Other configuration and malformed teams are not choices.
            relative = path.relative_to(root)
            category = relative.parent.as_posix()
            key = relative.as_posix()
            files[key] = path
            groups.setdefault(category, []).append(dict(
                id=key, name=team["name"], file=path.name, size=len(team["party"]),
                species=[mon["species"] for mon in team["party"]]))
        self.team_files = files
        return [dict(id=category, label="Misc" if category == "." else category,
                     teams=sorted(entries, key=lambda row: (row["name"].casefold(), row["id"])))
                for category, entries in sorted(groups.items())]

    def select(self, key):
        files = self.team_files  # A concurrent refresh replaces the index atomically.
        if not isinstance(key, str) or key not in files:
            raise ValueError("select an available team; refresh the team list")
        path = files[key]
        try:
            team = self.read_team(path)
        except (ValueError, TypeError, OSError, RuntimeError) as error:
            raise ValueError(f"team is unavailable; refresh the team list: {error}") from error
        category = path.relative_to(self.project / "config").parent.as_posix()
        return dict(team=team, category="Misc" if category == "." else category, template=path.name)

    def sample(self, category, size):
        if category not in (*CATEGORIES, "mixed"):
            raise ValueError("unknown team category")
        positive_int(size, "party size", 6)
        chosen = random.SystemRandom().choice(CATEGORIES) if category == "mixed" else category
        directory = self.project / "config/teams/profiles" / chosen / str(size)
        files = sorted(directory.glob("*.json"))
        if not files:
            raise ValueError(f"no {chosen} templates for a party of {size}")
        path = random.SystemRandom().choice(files)
        value = json.loads(path.read_text())
        team = self.validate({"name": value.get("name", path.stem), "party": value.get("party")})
        return dict(team=team, category=chosen, template=path.stem)


@dataclass(frozen=True)
class PlaySettings:
    project: Path
    rom: Path
    source: Path
    emulator: Path
    checkpoint: Path | None = None
    checkpoint_dir: Path | None = None
    model_list: Path | None = None
    ports: tuple[int, int] = (55365, 55366)
    workers: int = 2
    speed: float | str = 1.0
    capture_speed: float | str = "unlimited"
    setup_speed: float | str = "unlimited"
    build_timeout: float = 180
    startup_timeout: float = 30
    connect_timeout: float = 180
    max_steps: int = 200
    device: str = "cpu"


class Cancelled(RuntimeError):
    pass


def tail(path, maximum=14000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - maximum))
            return stream.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


class MatchJobs:
    """One owned match at a time, with immutable inputs and cancellable child tools."""
    def __init__(self, settings, catalog):
        self.settings = settings
        self.catalog = catalog
        self.lock = threading.RLock()
        self.checkpoints = {}
        self.job = None
        self.thread = None
        self.cancelled = threading.Event()
        self.process = None
        self.closed = False
        if settings.model_list is not None:
            if settings.checkpoint is not None:
                raise ValueError("--model-list cannot be combined with --checkpoint")
            self.models()  # Fail before opening the browser if classroom choices are unavailable.

    def _listed_models(self):
        cfg = self.settings
        if cfg.checkpoint is not None:
            raise ValueError("--model-list cannot be combined with --checkpoint")
        manifest = cfg.model_list.expanduser()
        if not manifest.is_absolute():
            manifest = cfg.project / manifest
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read --model-list {manifest}: {error}") from error
        if (not isinstance(data, dict) or set(data) != {"version", "default", "models"}
                or type(data.get("version")) is not int or data["version"] != 1):
            raise ValueError("--model-list needs exactly version: 1, default, and models")
        if not isinstance(data["default"], str) or not data["default"].strip():
            raise ValueError("--model-list default must be a nonempty model id")
        if not isinstance(data["models"], list) or not data["models"]:
            raise ValueError("--model-list models must be a nonempty list")
        found, rows = {}, []
        for entry in data["models"]:
            if not isinstance(entry, dict) or set(entry) != {"id", "label", "checkpoint"}:
                raise ValueError("each --model-list entry needs exactly id, label, and checkpoint")
            if any(not isinstance(entry[key], str) or not entry[key].strip()
                   for key in ("id", "label", "checkpoint")):
                raise ValueError("model id, label, and checkpoint must be nonempty text")
            key = entry["id"]
            if key in found:
                raise ValueError(f"duplicate --model-list id: {key}")
            path = Path(entry["checkpoint"]).expanduser()
            if not path.is_absolute():
                path = cfg.project / path
            path = path.resolve()
            if path.suffix.lower() != ".zip" or not path.is_file():
                raise ValueError(f"model {key!r} checkpoint must be an existing .zip file: {path}")
            found[key] = path
            rows.append(dict(id=key, label=entry["label"].strip()))
        if data["default"] not in found:
            raise ValueError(f"--model-list default id is not listed: {data['default']}")
        rows.sort(key=lambda row: row["id"] != data["default"])
        with self.lock:
            self.checkpoints = found
        return rows

    def models(self):
        cfg = self.settings
        if cfg.model_list is not None:
            return self._listed_models()
        # Named models are always listed, independently of the recent-run limit.
        model_root = cfg.project / "models"
        named = sorted((path for path in model_root.rglob("*.zip") if path.is_file()),
                       key=lambda path: (len(path.relative_to(model_root).parts),
                                         path.as_posix().casefold(), path.as_posix()))
        paths = list(named)
        if cfg.checkpoint_dir and cfg.checkpoint_dir.is_dir():
            dated = []
            for path in cfg.checkpoint_dir.rglob("*.zip"):
                try:
                    dated.append((path.stat().st_mtime, path))
                except FileNotFoundError:
                    continue  # A running trainer can replace a checkpoint during refresh.
            paths.extend(p for _, p in sorted(dated, reverse=True)[:200])
        if cfg.checkpoint and not is_latest(cfg.checkpoint):
            paths.insert(0, cfg.checkpoint)
        elif not cfg.checkpoint:
            preferred = next((path for path in named
                              if path.stem.casefold().replace("_", "").replace("-", "").replace(" ", "") == "smartv10"), None)
            if preferred is not None:
                paths.insert(0, preferred)
        found, rows = {}, []
        for path in paths:
            path = path.resolve()
            if not path.is_file():
                continue
            key = hashlib.sha256(str(path).encode()).hexdigest()[:24]
            if key in found:
                continue
            found[key] = path
            try:
                label = str(path.relative_to(cfg.project))
            except ValueError:
                label = str(path)
            rows.append(dict(id=key, label=label))
        try:
            resolve_checkpoint("latest", cfg.project, search_root=cfg.checkpoint_dir)
        except ValueError:
            pass
        else:
            found["latest"] = Path("latest")
            entry = dict(id="latest", label="Latest saved checkpoint (resolved when match starts)")
            if cfg.checkpoint and is_latest(cfg.checkpoint):
                rows.insert(0, entry)
            else:
                rows.append(entry)
        with self.lock:
            self.checkpoints = found
        return rows

    def describe(self):
        return dict(workers=min(self.settings.workers, 2), speed=self.settings.speed,
                    capture_speed=self.settings.capture_speed, setup_speed=self.settings.setup_speed, checkpoints=self.models(),
                    ports=self.settings.ports)

    def start(self, payload):
        if not isinstance(payload, dict) or set(payload) - {"teams", "checkpoint", "workers", "speed", "stochastic", "player_name"}:
            raise ValueError("invalid match settings")
        teams = payload.get("teams")
        if not isinstance(teams, list) or len(teams) != 2:
            raise ValueError("provide your team and the AI team")
        teams = [self.catalog.validate(team) for team in teams]
        player_name = payload.get("player_name", "")
        if not isinstance(player_name, str):
            raise ValueError("player name must be text")
        player_name = player_name.strip()
        if player_name:
            if len(player_name) > 7 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 " for c in player_name):
                raise ValueError("player name: use 1–7 letters, digits, or spaces")
            teams[0]["trainer_name"] = player_name
        workers = positive_int(payload.get("workers", min(self.settings.workers, 2)), "workers", min(self.settings.workers, 2))
        try:
            speed = speed_value(str(payload.get("speed", self.settings.speed)))
        except argparse.ArgumentTypeError as error:
            raise ValueError(str(error)) from error
        stochastic = payload.get("stochastic", False)
        if type(stochastic) is not bool:
            raise ValueError("stochastic must be true or false")
        with self.lock:
            if self.closed:
                raise RuntimeError("the local server is shutting down")
            if self.thread and self.thread.is_alive():
                raise RuntimeError("a match is already running; stop it before starting another")
            key = payload.get("checkpoint")
            if not isinstance(key, str) or key not in self.checkpoints:
                raise ValueError("select an available PPO checkpoint")
            checkpoint = resolve_checkpoint(self.checkpoints[key], self.settings.project,
                                            search_root=self.settings.checkpoint_dir)
            if not checkpoint.is_file():
                raise ValueError("checkpoint no longer exists; refresh the checkpoint list")
            run = self.settings.project / ".local/run" / (datetime.now(timezone.utc).strftime("play-ui-%Y%m%d-%H%M%S-") + secrets.token_hex(4))
            run.mkdir(parents=True)
            self.cancelled = threading.Event()
            self.job = dict(id=run.name, status="preparing", message="Preparing your match…",
                            directory=str(run), workers=workers, teams=teams, result=None, error=None)
            self.thread = threading.Thread(target=self._work, args=(run, teams, checkpoint, workers, speed, stochastic),
                                           name="human-match", daemon=False)
            self.thread.start()
        return self.snapshot()

    def update(self, **fields):
        with self.lock:
            self.job.update(fields)

    def snapshot(self):
        with self.lock:
            if self.job is None:
                return None
            state = copy.deepcopy(self.job)
        run = Path(state["directory"])
        state["generation_log"] = tail(run / "generation.log")
        state["battle_log"] = tail(run / "battle-launch.log")
        if state["status"] == "connecting" and "READY: you are player" in state["battle_log"]:
            state.update(status="playing", message="Your battle is ready. Play in the Player 1 mGBA window.")
            with self.lock:
                if self.job["status"] == "connecting":
                    self.job.update(status=state["status"], message=state["message"])
        state["states_ready"] = sum(p.is_file() for p in (run / "states/counter/player1.ss1", run / "states/counter/player2.ss1"))
        return state

    def cancel(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                self.cancelled.set()
                self.job.update(status="stopping", message="Stopping the match and its emulators…")
        return self.snapshot()

    def shutdown(self):
        with self.lock:
            self.closed = True
        self.cancel()
        if self.thread:
            self.thread.join()

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise Cancelled("Match stopped.")

    def _run(self, command, log_path):
        self.check_cancelled()
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, cwd=self.settings.project, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with self.lock:
                self.process = process
            interrupted = None
            child_groups = set()
            try:
                while process.poll() is None:
                    if self.cancelled.is_set():
                        if interrupted is None:
                            # The Python entry points stop their native mGBA children in finally blocks.
                            child_groups = descendant_groups(process.pid)
                            signal_group(process.pid, signal.SIGINT)
                            interrupted = time.monotonic()
                        elif time.monotonic() - interrupted > 15:
                            signal_group(process.pid, signal.SIGKILL)
                    time.sleep(0.1)
                self.check_cancelled()
                if process.returncode:
                    detail = tail(log_path, 3000).strip()
                    raise RuntimeError(detail or f"{Path(command[2]).name} exited with code {process.returncode}")
            finally:
                if process.poll() is None:
                    child_groups |= descendant_groups(process.pid)
                    signal_group(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        signal_group(process.pid, signal.SIGKILL)
                        process.wait()
                # mGBA and Xvfb create separate sessions. Clean up known descendants
                # if the Python child could not finish its normal cleanup.
                for group in child_groups:
                    signal_group(group, signal.SIGKILL)
                with self.lock:
                    self.process = None

    def _work(self, run, teams, checkpoint, workers, speed, stochastic):
        cfg = self.settings
        try:
            team_paths = [run / "player1.json", run / "player2.json"]
            for team, path in zip(teams, team_paths, strict=True):
                path.write_text(json.dumps(team, indent=2) + "\n", encoding="utf-8")
            # Freeze this match's policy even when training updates latest.zip concurrently.
            data = checkpoint.read_bytes()
            agent = run / "agent.zip"
            agent.write_bytes(data)
            (run / "config.json").write_text(json.dumps(dict(checkpoint=str(checkpoint),
                checkpoint_sha256=hashlib.sha256(data).hexdigest(), workers=workers,
                teams=[str(p) for p in team_paths], speed=speed, capture_speed=cfg.capture_speed, setup_speed=cfg.setup_speed,
                ports=cfg.ports, stochastic=stochastic), indent=2) + "\n")
            self.check_cancelled()
            self.update(status="generating", message=f"Generating and verifying two teams with {workers} worker(s)…")
            command = [sys.executable, "-u", str(cfg.project / "mgba/build_states.py"),
                       "--source", str(cfg.source), "--rom", str(cfg.rom), "--emulator", str(cfg.emulator),
                       "--teams", *map(str, team_paths), "--workers", str(workers),
                       "--output-dir", str(run / "states"), "--headless", "--speed", str(cfg.capture_speed),
                       "--timeout", str(cfg.build_timeout)]
            self._run(command, run / "generation.log")
            states = [run / f"states/counter/player{i}.ss1" for i in (1, 2)]
            for path in states:
                report = json.loads(path.with_suffix(".ss1.json").read_text())
                if not report.get("ok") or not path.is_file() or not path.stat().st_size:
                    raise RuntimeError(f"state verification failed: {path.name}")
                if report.get("state_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise RuntimeError(f"state checksum differs from generation report: {path.name}")
            self.check_cancelled()
            self.update(status="connecting", message="Saves are ready. Opening mGBA and connecting the link battle…")
            command = [sys.executable, "-u", str(cfg.project / "mgba/play_agent.py"),
                       "--rom", str(cfg.rom), "--emulator", str(cfg.emulator), "--checkpoint", str(agent),
                       "--states", *map(str, states), "--human-player", "1", "--ports", *map(str, cfg.ports),
                       "--speed", str(speed), "--device", cfg.device, "--max-steps", str(cfg.max_steps),
                       "--setup-speed", str(cfg.setup_speed),
                       "--startup-timeout", str(cfg.startup_timeout), "--connect-timeout", str(cfg.connect_timeout),
                       "--output-dir", str(run / "battle"), "--no-wait"]
            if stochastic:
                command.append("--stochastic")
            self._run(command, run / "battle-launch.log")
            result = json.loads((run / "battle/result.json").read_text())
            if result.get("status") == "interrupted":
                self.update(status="cancelled", message="Match stopped.", result=result)
            else:
                self.update(status="complete", message="Match complete.", result=result)
        except Cancelled:
            self.update(status="cancelled", message="Match stopped.")
        except Exception as error:
            self.update(status="failed", message="The match could not finish.", error=str(error))
        finally:
            (run / "job.json").write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")


def signal_group(group, signum):
    try:
        os.killpg(group, signum)
    except ProcessLookupError:
        pass


def descendant_groups(pid):
    """Collect only this launched tool's descendants on the supported Linux host."""
    relations = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            # The process name is parenthesized and may itself contain spaces.
            fields = path.read_text().rsplit(")", 1)[1].split()
            relations[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {pid}
    while True:
        found = {child for child, parent in relations.items() if parent in descendants}
        if found <= descendants:
            break
        descendants |= found
    groups = set()
    for child in descendants:
        try:
            group = os.getpgid(child)
            if group != os.getpgrp():
                groups.add(group)
        except ProcessLookupError:
            pass
    return groups


class PlayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, jobs, catalog):
        self.jobs, self.catalog = jobs, catalog
        self.token = secrets.token_urlsafe(32)
        self.assets = jobs.settings.project / "web/play"
        super().__init__(("127.0.0.1", port), PlayHandler)
        self.origins = {f"http://127.0.0.1:{self.server_port}", f"http://localhost:{self.server_port}"}


class PlayHandler(BaseHTTPRequestHandler):
    server: PlayServer

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *args):
        pass

    def send(self, status, data, content_type="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        if f"http://{self.headers.get('Host', '')}" not in self.server.origins:
            self.send(403, {"error": "use the local URL printed by the launcher"})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        path = unquote(urlsplit(self.path).path)
        if path == "/api/catalog":
            self.send(200, dict(catalog=self.server.catalog.public(), settings=self.server.jobs.describe(), token=self.server.token))
        elif path == "/api/job":
            self.send(200, {"job": self.server.jobs.snapshot()})
        elif path == "/api/teams":
            self.send(200, {"categories": self.server.catalog.available_teams()})
        else:
            static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            relative = static.get(path)
            if path.startswith("/sprites/") and path.endswith(".png"):
                number = path.removeprefix("/sprites/").removesuffix(".png")
                if number.isascii() and number.isdigit() and 1 <= int(number) <= 386:
                    relative = f"sprites/{int(number)}.png"
            if not relative:
                self.send(404, {"error": "not found"})
                return
            try:
                file = self.server.assets / relative
                self.send(200, file.read_bytes(), mimetypes.guess_type(str(file))[0] or "application/octet-stream")
            except OSError:
                self.send(404, {"error": "asset not found"})

    def do_POST(self):
        if not self.local_request():
            return
        origin = self.headers.get("Origin")
        if (origin is not None and origin not in self.server.origins) or self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.send(403, {"error": "local browser requests only"})
            return
        if not secrets.compare_digest(self.headers.get("X-Play-Token", ""), self.server.token):
            self.send(403, {"error": "refresh the page before submitting"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536 or self.headers.get_content_type() != "application/json":
                raise ValueError("send a JSON body of at most 64 KiB")
            payload = json.loads(self.rfile.read(length))
            path = urlsplit(self.path).path
            if path == "/api/team":
                if isinstance(payload, dict) and set(payload) == {"id"}:
                    self.send(200, self.server.catalog.select(payload["id"]))
                elif isinstance(payload, dict) and set(payload) == {"category", "size"}:
                    self.send(200, self.server.catalog.sample(payload["category"], payload["size"]))
                else:
                    raise ValueError("provide a saved team id, or category and size for a random team")
            elif path == "/api/jobs":
                self.send(202, {"job": self.server.jobs.start(payload)})
            elif path == "/api/job/cancel":
                self.send(200, {"job": self.server.jobs.cancel()})
            else:
                self.send(404, {"error": "not found"})
        except (ValueError, TypeError, OSError) as error:
            self.send(400, {"error": str(error)})
        except RuntimeError as error:
            self.send(409, {"error": str(error)})
