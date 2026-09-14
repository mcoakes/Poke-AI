from enum import IntEnum


class GBAAction(IntEnum):
    NOOP = 0
    A = 1
    B = 2
    UP = 3
    DOWN = 4
    LEFT = 5
    RIGHT = 6
    START = 7
    SELECT = 8
    L = 9
    R = 10


ACTION_COUNT = len(GBAAction)

