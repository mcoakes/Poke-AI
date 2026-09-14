"""Choose counter checkpoints reproducibly without sharing RNGs between workers."""
import hashlib
import json
from pathlib import Path
import random
import re

from .team_profiles import add_profile_arguments, profile_files, profile_location, validate_profile


def add_state_arguments(parser, *, include_teams=False, required=True):
    selection = parser.add_mutually_exclusive_group(required=required)
    selection.add_argument("--states", nargs=2, type=Path, help="Two fixed counter states, in player order")
    selection.add_argument("--state-dir", type=Path, help="Draw two random .ssN counter states from this directory")
    if include_teams:
        selection.add_argument("--teams", nargs=2, type=Path, help="Showdown: two fixed team JSON files, in player order")
        selection.add_argument("--team-dir", type=Path, help="Showdown: draw random team JSON files from this directory")
        selection.add_argument("--training-profile", help="Showdown: named team directory under config/training_profiles/NAME/teams")
    add_profile_arguments(parser, selection)
    parser.add_argument("--mirror-teams", action="store_true",
                        help="Draw one random team/state per battle and give it to both players")
    return selection


class TrainingStates:
    def __init__(self, *, states=None, directory=None, profile=None, team_size=None,
                 profile_root=None, seed=1, mirror_teams=False):
        validate_profile(profile, team_size)
        if sum(value is not None for value in (states, directory, profile)) != 1:
            raise ValueError("choose exactly one of --states FILE FILE, --state-dir DIRECTORY, or --team-profile PROFILE")
        self.fixed = states is not None
        self.mirror_teams = bool(mirror_teams)
        if self.fixed and self.mirror_teams:
            raise ValueError("--mirror-teams requires a random directory or --team-profile; for a fixed mirror pass the same file twice")
        self.profile, self.team_size = profile, team_size
        self.groups = None
        if self.fixed:
            if len(states) != 2:
                raise ValueError("--states requires exactly two counter checkpoints")
            files = [Path(p).expanduser().resolve() for p in states]
        elif profile is not None:
            self.profile_root = Path(profile_root or Path(__file__).resolve().parents[2] / "states/counter/profiles").expanduser().resolve()
            self.groups = profile_files(self.profile_root, profile, team_size, states=True)
            files = [p for group in self.groups.values() for p in group]
            for path in files:
                category, size = profile_location(path, self.profile_root)
                manifest = path.with_name(path.name + ".json")
                if manifest.is_file():
                    data = json.loads(manifest.read_text())
                    if (data.get("team_profile", category) != category or data.get("team_size", size) != size
                            or ("verified_party" in data and len(data["verified_party"]) != size)):
                        raise ValueError(f"checkpoint metadata disagrees with its category/size directory: {path}")
        else:
            directory = Path(directory).expanduser().resolve()
            if not directory.is_dir():
                raise ValueError(f"state directory does not exist: {directory}")
            files = sorted(p.resolve() for p in directory.iterdir()
                           if p.is_file() and re.fullmatch(r"\.ss\d+", p.suffix.lower()))
            if not files:
                raise ValueError(f"no .ss1 (or other .ssN) checkpoints in {directory}")
        for path in files:
            if not path.is_file():
                raise ValueError(f"checkpoint does not exist: {path}")
            if not re.fullmatch(r"\.ss\d+", path.suffix.lower()):
                raise ValueError(f"expected a counter checkpoint (.ss1), not a battery save: {path}")
        self.files, self.seed = tuple(files), seed
        self.hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.files}

    @classmethod
    def from_args(cls, args, project):
        return cls(states=args.states, directory=args.state_dir, profile=args.team_profile,
                   team_size=args.team_size, profile_root=project / "states/counter/profiles", seed=args.seed,
                   mirror_teams=getattr(args, "mirror_teams", False))

    @property
    def label(self):
        if self.profile:
            return f"{self.profile} teams, size {self.team_size or '1–6'}" + (" (mirrored)" if self.mirror_teams else "")
        return "fixed pair" if self.fixed else "random directory" + (" (mirrored)" if self.mirror_teams else "")

    def for_pair(self, index):
        rng = random.Random(self.seed + index * 1000003)

        def choose():
            if self.fixed:
                return self.files
            if self.mirror_teams:
                pool = self.groups[rng.choice(tuple(self.groups))] if self.groups else self.files
                selected = rng.choice(pool)
                return selected, selected
            if self.groups is not None:
                # Mixed weights categories equally, even if different numbers of states were built.
                first_group, second_group = (rng.choice(tuple(self.groups)) for _ in range(2))
                first = rng.choice(self.groups[first_group])
                second_pool = self.groups[second_group]
                alternatives = tuple(p for p in second_pool if p != first)
                return first, rng.choice(alternatives or second_pool)
            if len(self.files) == 1:
                return (self.files[0], self.files[0])
            return tuple(rng.sample(self.files, 2))
        return choose

    def describe(self):
        result = dict(mode="fixed" if self.fixed else "random_directory", seed=self.seed,
                      mirror_teams=self.mirror_teams,
                      checkpoints=[dict(path=str(p), sha256=self.hashes[str(p)]) for p in self.files])
        if self.profile:
            result.update(mode="team_profile", team_profile=self.profile, team_size=self.team_size,
                          root=str(self.profile_root), category_sampling="uniform_per_battle" if self.mirror_teams else "uniform_per_player",
                          category_counts={key: len(paths) for key, paths in self.groups.items()})
        return result
