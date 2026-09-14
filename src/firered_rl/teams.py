from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import subprocess

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
NATURES = ("Hardy", "Lonely", "Brave", "Adamant", "Naughty", "Bold", "Docile",
           "Relaxed", "Impish", "Lax", "Timid", "Hasty", "Serious", "Jolly", "Naive",
           "Modest", "Mild", "Quiet", "Bashful", "Rash", "Calm", "Gentle", "Sassy",
           "Careful", "Quirky")


def stat_values(value, *, default, maximum, label):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(STATS):
        raise ValueError(f"{label}: use stat keys {', '.join(STATS)}")
    values = tuple(value.get(stat, default) for stat in STATS)
    if any(type(n) is not int or not 0 <= n <= maximum for n in values):
        raise ValueError(f"{label}: each value must be an integer between 0 and {maximum}")
    if label == "evs" and sum(values) > 510:
        raise ValueError("evs: total cannot exceed 510")
    return values


@dataclass(frozen=True)
class PokemonSpec:
    species: str
    level: int
    moves: tuple[str, ...]
    evs: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    ivs: tuple[int, ...] = (31, 31, 31, 31, 31, 31)
    held_item: str = "None"
    nature: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "PokemonSpec":
        if not isinstance(value, dict):
            raise ValueError("each Pokemon must be an object")
        unknown = set(value) - {"species", "level", "moves", "evs", "ivs", "held_item", "nature"}
        if unknown:
            raise ValueError(f"unknown Pokemon fields: {', '.join(sorted(unknown))}")
        species = value.get("species", "")
        level = value.get("level", 0)
        raw_moves = value.get("moves", ())
        if not isinstance(species, str) or not species.strip():
            raise ValueError("Pokemon species cannot be empty")
        species = species.strip()
        if type(level) is not int or not 1 <= level <= 100:
            raise ValueError(f"{species}: level must be between 1 and 100")
        if not isinstance(raw_moves, (list, tuple)) or not 1 <= len(raw_moves) <= 4 or any(
                not isinstance(move, str) or not move.strip() for move in raw_moves):
            raise ValueError(f"{species}: provide between one and four moves")
        moves = tuple(move.strip() for move in raw_moves)
        if len(set(move.lower().replace(' ', '') for move in moves)) != len(moves):
            raise ValueError(f"{species}: duplicate moves")
        item = value.get("held_item", "None")
        if not isinstance(item, str) or not item.strip():
            raise ValueError("held_item must be an item name or 'None'")
        nature = value.get("nature")
        if nature is not None:
            if not isinstance(nature, str) or nature.title() not in NATURES:
                raise ValueError("unknown nature")
            nature = nature.title()
        return cls(species=species, level=level, moves=moves,
                   evs=stat_values(value.get("evs"), default=0, maximum=255, label="evs"),
                   ivs=stat_values(value.get("ivs"), default=31, maximum=31, label="ivs"),
                   held_item=item.strip(), nature=nature)


@dataclass(frozen=True)
class TeamSpec:
    name: str
    party: tuple[PokemonSpec, ...]

    @classmethod
    def load(cls, path: str | Path) -> "TeamSpec":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        name = str(value.get("name", source.stem)).strip()
        party = tuple(PokemonSpec.from_dict(item) for item in value.get("party", ()))
        if not name:
            raise ValueError("team name cannot be empty")
        if not 1 <= len(party) <= 6:
            raise ValueError("a team must contain between one and six Pokemon")
        return cls(name=name, party=party)


class TeamPool:
    """Select reproducible team specifications for training resets."""

    def __init__(self, paths: list[str | Path]) -> None:
        if not paths:
            raise ValueError("team pool cannot be empty")
        self.paths = tuple(Path(path).resolve() for path in paths)
        for path in self.paths:
            TeamSpec.load(path)

    def choose(self, seed: int | None = None) -> Path:
        return random.Random(seed).choice(self.paths)


class PKHeXTeamEditor:
    """Generate a FireRed save by invoking the pinned PKHeX.Core sidecar."""

    def __init__(self, project: str | Path, dotnet: str = "dotnet") -> None:
        self.project = Path(project).resolve()
        self.dotnet = dotnet

    def build(
        self,
        template_save: str | Path,
        team_file: str | Path,
        output_save: str | Path,
    ) -> Path:
        template = Path(template_save).resolve()
        team = Path(team_file).resolve()
        output = Path(output_save).resolve()
        TeamSpec.load(team)
        if not template.is_file():
            raise FileNotFoundError(f"template save does not exist: {template}")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.dotnet,
            "run",
            "--project",
            str(self.project),
            "--",
            "--template",
            str(template),
            "--team",
            str(team),
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"PKHeX team editor failed: {detail}")
        if not output.is_file():
            raise RuntimeError("PKHeX team editor did not create the requested save")
        return output
