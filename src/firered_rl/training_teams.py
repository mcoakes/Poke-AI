"""JSON team selection for the simulator, using the same profile sampling as mGBA."""
import hashlib
from pathlib import Path

from .teams import TeamSpec
from .team_profiles import profile_files, validate_profile
from .training_states import TrainingStates
from .training_profiles import training_profile_directory


class TrainingTeams(TrainingStates):
    def __init__(self, *, teams=None, directory=None, profile=None, team_size=None,
                 profile_root=None, seed=1, mirror_teams=False):
        validate_profile(profile, team_size)
        if sum(v is not None for v in (teams, directory, profile)) != 1:
            raise ValueError("Showdown requires --teams FILE FILE, --team-dir DIRECTORY, or --team-profile PROFILE")
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
                   mirror_teams=getattr(args, "mirror_teams", False))

    def describe(self):
        result = super().describe()
        result["teams"] = result.pop("checkpoints")
        return result
