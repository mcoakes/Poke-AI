"""Shared category/size layout for JSON templates and generated counter states."""
from pathlib import Path
import re


CATEGORIES = ("competitive", "casual", "random")
PROFILE_CHOICES = (*CATEGORIES, "mixed")


def validate_profile(profile, size):
    if size is not None and profile is None:
        raise ValueError("--team-size requires --team-profile; omit it with --states or --state-dir")
    if profile is not None and profile not in PROFILE_CHOICES:
        raise ValueError(f"unknown team profile: {profile}")
    if size is not None and (type(size) is not int or size not in range(1, 7)):
        raise ValueError("--team-size must be between 1 and 6")


def add_profile_arguments(parser, selection):
    selection.add_argument("--team-profile", choices=PROFILE_CHOICES,
                           help="Select competitive, casual, random, or a mixture of all three")
    parser.add_argument("--team-size", type=int, choices=range(1, 7),
                        help="Restrict both players to this party size; requires --team-profile")


def profile_location(path, root):
    parts = Path(path).relative_to(root).parts
    if len(parts) != 3 or parts[0] not in CATEGORIES or parts[1] not in "123456" or len(parts[1]) != 1:
        raise ValueError(f"expected CATEGORY/SIZE/FILE under {root}: {path}")
    return parts[0], int(parts[1])


def profile_files(root, profile, size=None, *, states=False):
    """Only inspect the requested category/size directories, never legacy siblings."""
    validate_profile(profile, size)
    if profile is None:
        raise ValueError("a team profile is required")
    root = Path(root).expanduser().resolve()
    groups = {}
    for category in CATEGORIES if profile == "mixed" else (profile,):
        files = []
        for count in (size,) if size is not None else range(1, 7):
            directory = root / category / str(count)
            if directory.is_dir():
                files.extend(p.resolve() for p in sorted(directory.iterdir()) if p.is_file()
                             and (re.fullmatch(r"\.ss\d+", p.suffix.lower()) if states else p.suffix == ".json"))
        if not files:
            where = root / category / (str(size) if size else "")
            kind = "counter states (.ssN)" if states else "team JSON templates"
            hint = (f" Build them first with mgba/build_states.py --team-profile {profile}"
                    + (f" --team-size {size}" if size else "") + " (plus --source and --rom).") if states else ""
            raise ValueError(f"no {kind} in {where}.{hint}")
        groups[category] = tuple(files)
    return groups
