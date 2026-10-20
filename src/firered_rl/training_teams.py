"""JSON team selection for the simulator, using the same profile sampling as mGBA."""
import hashlib
import json
import math
from pathlib import Path
import random
from threading import Lock

from .teams import TeamSpec
from .team_profiles import profile_files, validate_profile
from .training_states import TrainingStates
from .training_profiles import training_profile_directory


class TrainingTeams(TrainingStates):
    def __init__(self, *, teams=None, directory=None, profile=None, team_size=None,
                 profile_root=None, seed=1, mirror_teams=False, mixture=None):
        validate_profile(profile, team_size)
        if sum(v is not None for v in (teams, directory, profile, mixture)) != 1:
            raise ValueError("Showdown requires exactly one of --teams FILE FILE, --team-dir DIRECTORY, --team-profile PROFILE, or --team-mixture FILE")
        self.mixture = None
        if mixture is not None:
            if mirror_teams:
                raise ValueError("--mirror-teams cannot be combined with --team-mixture; use an explicit fixed pair in the manifest")
            self._load_mixture(mixture, seed)
            return
        self.fixed = teams is not None
        self.mirror_teams = bool(mirror_teams)
        if self.fixed and self.mirror_teams:
            raise ValueError("--mirror-teams requires a random directory or --team-profile; for a fixed mirror pass the same file twice")
        self.profile, self.team_size, self.seed = profile, team_size, seed
        self.groups = None
        if self.fixed:
            if len(teams) != 2:
                raise ValueError("--teams requires exactly two team JSON files")
            files = [Path(p).expanduser().resolve() for p in teams]
        elif profile:
            self.profile_root = Path(profile_root).expanduser().resolve()
            self.groups = profile_files(self.profile_root, profile, team_size)
            files = [p for group in self.groups.values() for p in group]
        else:
            directory = Path(directory).expanduser().resolve()
            if not directory.is_dir():
                raise ValueError(f"team directory does not exist: {directory}")
            files = sorted(directory.glob("*.json"))
        if not files:
            raise ValueError("no team JSON files found")
        for path in files:
            if path.suffix.lower() != ".json":
                raise ValueError(f"Showdown uses team JSON files, not emulator save states: {path}")
            team = TeamSpec.load(path)
            if profile and len(team.party) != int(path.parent.name):
                raise ValueError(f"team size disagrees with directory: {path}")
        self.files = tuple(files)
        self.hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

    @classmethod
    def from_args(cls, args, project):
        if args.states is not None or args.state_dir is not None:
            raise ValueError("Showdown reads JSON teams: use --teams, --team-dir, or --team-profile")
        directory = (training_profile_directory(project, args.training_profile)
                     if getattr(args, "training_profile", None) else args.team_dir)
        return cls(teams=args.teams, directory=directory, profile=args.team_profile,
                   team_size=args.team_size, profile_root=project / "config/teams/profiles", seed=args.seed,
                   mirror_teams=getattr(args, "mirror_teams", False), mixture=getattr(args, "team_mixture", None))

    def _load_mixture(self, path, seed):
        self.mixture = Path(path).expanduser().resolve()
        raw = self.mixture.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("team mixture requires integer version 1")
        if set(data) != {"version", "groups"}:
            raise ValueError("team mixture only accepts version and groups")
        groups = data["groups"]
        if not isinstance(groups, dict) or not groups:
            raise ValueError("team mixture groups must be a nonempty object")
        self.mixture_groups = {}
        for name, entry in groups.items():
            if not isinstance(name, str) or not name or not isinstance(entry, dict):
                raise ValueError("team mixture groups require nonempty names and objects")
            if set(entry) not in ({"weight", "directory"}, {"weight", "teams"}):
                raise ValueError(f"team mixture group {name}: choose weight plus directory OR teams")
            weight = entry["weight"]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"team mixture group {name}: weight must be positive and finite")

            def resolve(value):
                if not isinstance(value, str) or not value:
                    raise ValueError(f"team mixture group {name}: paths must be nonempty strings")
                return (self.mixture.parent / Path(value).expanduser()).resolve()

            if "directory" in entry:
                directory = resolve(entry["directory"])
                if not directory.is_dir():
                    raise ValueError(f"team mixture directory does not exist: {directory}")
                files = tuple(sorted(directory.glob("*.json")))
                selector = dict(directory=str(directory))
                fixed = False
            else:
                if not isinstance(entry["teams"], list) or len(entry["teams"]) != 2:
                    raise ValueError(f"team mixture group {name}: teams must contain exactly two paths")
                files = tuple(resolve(value) for value in entry["teams"])
                selector = dict(teams=[str(file) for file in files])
                fixed = True
            if not files:
                raise ValueError(f"team mixture group {name}: no team JSON files found")
            self.mixture_groups[name] = dict(weight=float(weight), files=files, fixed=fixed, **selector)
        self.files = tuple(dict.fromkeys(file for group in self.mixture_groups.values() for file in group["files"]))
        for file in self.files:
            if file.suffix.lower() != ".json":
                raise ValueError(f"team mixture requires team JSON files: {file}")
            TeamSpec.load(file)
        self.hashes = {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in self.files}
        self.manifest_hash = hashlib.sha256(raw).hexdigest()
        total = math.fsum(group["weight"] for group in self.mixture_groups.values())
        if not math.isfinite(total):
            raise ValueError("team mixture total weight must be finite")
        self.weights = {name: group["weight"] / total for name, group in self.mixture_groups.items()}
        self.sampling_counts = dict.fromkeys(self.mixture_groups, 0)
        self.count_lock = Lock()
        self.fixed, self.mirror_teams, self.groups = False, False, None
        self.profile, self.team_size, self.seed = None, None, seed

    @property
    def label(self):
        if self.mixture is not None:
            return "episode team mixture (" + ", ".join(f"{name}={weight:.0%}" for name, weight in self.weights.items()) + ")"
        return super().label

    def for_pair(self, index):
        if self.mixture is None:
            return super().for_pair(index)
        rng = random.Random(self.seed + index * 1000003)
        names, weights = tuple(self.weights), tuple(self.weights.values())

        def choose():
            name = rng.choices(names, weights=weights, k=1)[0]
            group = self.mixture_groups[name]
            files = group["files"]
            with self.count_lock:
                self.sampling_counts[name] += 1
            if group["fixed"]:
                return files
            if len(files) == 1:
                return files[0], files[0]
            return tuple(rng.sample(files, 2))
        return choose

    def sampling_summary(self):
        with self.count_lock:
            counts = dict(self.sampling_counts)
        return dict(weights=self.weights, episode_selections=counts, total=sum(counts.values()),
                    counting="episode starts, including unfinished episodes; current invocation only")

    def snapshot(self, directory, *, expected=None):
        """Train from frozen copies, retaining original fingerprints for continuation validation."""
        directory = Path(directory).resolve()
        manifest = directory / "mixture.json"
        if expected is not None:
            frozen = TrainingTeams(mixture=manifest, seed=self.seed)
            if frozen.describe() != expected:
                raise ValueError("pinned team mixture inputs changed; start a new run with --resume")
            return frozen
        directory.mkdir(parents=True, exist_ok=False)
        raw = self.mixture.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest_hash:
            raise ValueError("team mixture manifest changed during preflight")
        (directory / "original-manifest.json").write_bytes(raw)
        groups = {}
        for index, (name, group) in enumerate(self.mixture_groups.items()):
            target_dir = directory / f"group-{index:03d}"
            target_dir.mkdir()
            copied = []
            for position, file in enumerate(group["files"]):
                raw = file.read_bytes()
                if hashlib.sha256(raw).hexdigest() != self.hashes[str(file)]:
                    raise ValueError(f"team input changed during preflight: {file}")
                target = target_dir / f"team-{position:05d}.json"
                target.write_bytes(raw)
                copied.append(str(target.relative_to(directory)))
            selector = {"teams": copied} if group["fixed"] else {"directory": str(target_dir.relative_to(directory))}
            groups[name] = dict(weight=group["weight"], **selector)
        manifest.write_text(json.dumps(dict(version=1, groups=groups), indent=2) + "\n")
        return TrainingTeams(mixture=manifest, seed=self.seed)

    def describe(self):
        result = super().describe()
        result["teams"] = result.pop("checkpoints")
        if self.mixture is not None:
            result.update(mode="team_mixture", manifest=dict(path=str(self.mixture), sha256=self.manifest_hash),
                          sampling="one_group_per_episode", groups={name: dict(weight=group["weight"],
                              probability=self.weights[name], fixed_pair=group["fixed"],
                              files=[str(file) for file in group["files"]],
                              **({"directory": group["directory"]} if not group["fixed"] else {}))
                              for name, group in self.mixture_groups.items()})
        return result
