"""Versioned battle observations alongside the frozen v1 schema.

Both sides use player/opponent order. Party arrays retain team-slot order;
stats are stored values before boosts (active stats include Transform).
"""
from collections.abc import Mapping
from functools import lru_cache

import gymnasium as gym
import numpy as np

from .turn_env import observation_space


V2_FIELDS = (
    "v2_party", "v2_stats", "v2_active_stats", "v2_active", "v2_boosts",
    "v2_move_details", "v2_weather", "v2_volatiles",
)


def validate_version(version):
    if version not in ("v1", "v2"):
        raise ValueError(f"unsupported observation version {version!r}; expected v1 or v2")
    return version


def _columns(shape, low, high):
    return gym.spaces.Box(np.broadcast_to(np.asarray(low, dtype=np.int64), shape).copy(),
                          np.broadcast_to(np.asarray(high, dtype=np.int64), shape).copy(),
                          dtype=np.int64)


def v2_spaces():
    """Raw additional fields with bounds for each semantic column."""
    return {
        # valid, level, current HP, max HP, types, item, ability, fainted, status
        "v2_party": _columns((2, 6, 10), 0,
                             [1, 100, 65535, 65535, 31, 31, 511, 255, 1, 2**32-1]),
        "v2_stats": gym.spaces.Box(0, 65535, (2, 6, 5), dtype=np.int64),
        "v2_active_stats": gym.spaces.Box(0, 65535, (2, 5), dtype=np.int64),
        "v2_active": _columns((2, 4), 0, [65535, 65535, 511, 255]),
        "v2_boosts": gym.spaces.Box(-6, 6, (2, 7), dtype=np.int64),
        # power, accuracy (0 for always-hit), always-hit, category, priority, type
        "v2_move_details": _columns((2, 6, 4, 6), [0, 0, 0, 0, -7, 0],
                                    [65535, 100, 1, 2, 7, 31]),
        # none/sun/rain/sand/hail, remaining turns, permanent
        "v2_weather": _columns((3,), 0, [4, 65535, 1]),
        "v2_volatiles": gym.spaces.Box(0, 2**32-1, (2,), dtype=np.uint32),
    }


def versioned_raw_space(version="v1"):
    validate_version(version)
    base = observation_space()
    if version == "v1":
        return base
    return gym.spaces.Dict({**base.spaces, **v2_spaces()})


def raw_observation_version(observation):
    return "v2" if any(key.startswith("v2_") for key in observation) else "v1"


@lru_cache(maxsize=1)
def _validation_specs():
    # Byte-backed arrays cannot be made writable; callers of public space
    # factories can freely mutate their Boxes without altering validation.
    specs = []
    for key, space in v2_spaces().items():
        low = np.frombuffer(space.low.tobytes(), dtype=space.dtype).reshape(space.shape)
        high = np.frombuffer(space.high.tobytes(), dtype=space.dtype).reshape(space.shape)
        specs.append((key, space.shape, low, high, space.dtype))
    return tuple(specs)


@lru_cache(maxsize=1)
def _base_validation_space():
    # Only this module uses this space, and only for contains checks.
    return observation_space()


def validate_v2_fields(fields):
    """Validate before casting, so fractions, NaNs, and uint32 wraps fail."""
    if not isinstance(fields, Mapping):
        raise ValueError("observation_v2 must contain native battle fields")
    expected, actual = set(V2_FIELDS), set(fields)
    if actual != expected:
        missing, extra = sorted(expected-actual), sorted(actual-expected)
        raise ValueError(f"invalid observation_v2 fields: missing={missing}, extra={extra}")
    result = {}
    for key, shape, low, high, dtype in _validation_specs():
        try:
            value = np.asarray(fields[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be a rectangular numeric array") from error
        if value.shape != shape:
            raise ValueError(f"{key} has shape {value.shape}; expected {shape}")
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite integer values")
        if (value < low).any() or (value > high).any():
            raise ValueError(f"{key} values fall outside the v2 field bounds")
        if (value != np.floor(value)).any():
            raise ValueError(f"{key} must contain integer values")
        result[key] = value.astype(dtype, copy=True)
    return result


def encode_v2(report, base_obs):
    """Add complete native fields; v2 cannot be reconstructed from v1 reports."""
    if report.get("backend") == "mgba":
        from .mgba_observation_v2 import native_fields
        extra = native_fields(report)
    elif report.get("backend") == "showdown":
        if "observation_v2" not in report:
            raise ValueError("Showdown report is missing native observation_v2 fields")
        extra = validate_v2_fields(report["observation_v2"])
    else:
        raise ValueError("observation v2 requires native Showdown or paired mGBA fields")
    base_space = _base_validation_space()
    if set(base_obs) != set(base_space.spaces):
        raise ValueError("encode_v2 requires a complete v1 base observation")
    if not base_space.contains(base_obs):
        raise ValueError("v1 base observation has invalid shapes, values, or dtypes")
    return {**{key: np.asarray(value).copy() for key, value in base_obs.items()}, **extra}
