"""Common pacing options for every entry point that starts mGBA."""
import argparse
import math


def speed_value(value):
    if value.lower() == "unlimited":
        return "unlimited"
    try:
        speed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("speed must be a positive multiplier or unlimited")
    if not math.isfinite(speed) or speed <= 0 or speed * 60 > 3.4e38:
        raise argparse.ArgumentTypeError("speed must be positive, finite, and representable by mGBA")
    return speed


def add_speed_argument(parser):
    parser.add_argument("--speed", type=speed_value, default="1",
                        help="Speed multiplier, e.g. 4, or unlimited (default: 1)")


def speed_options(speed):
    if speed == "unlimited":
        return ["-C", "audioSync=0", "-C", "videoSync=0"]
    fps = (16777216 / 280896) * speed
    return ["-C", f"fpsTarget={fps:.9g}", "-C", "audioSync=1", "-C", "videoSync=0"]
