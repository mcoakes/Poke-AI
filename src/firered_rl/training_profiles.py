"""Named, deliberately small JSON team directories for controlled training."""
from pathlib import Path
import re


def training_profile_directory(project, name):
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError("training profile names use lowercase letters, digits, _ and -")
    directory = Path(project) / "config/training_profiles" / name / "teams"
    if not directory.is_dir():
        raise ValueError(f"training profile not found: {directory}")
    return directory
