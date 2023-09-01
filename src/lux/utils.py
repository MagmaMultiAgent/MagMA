"""
This file contains utility functions that are used throughout the codebase.
"""

def my_turn_to_place_factory(place_first: bool, step: int):
    """
    Returns true if it is my turn to place a factory
    """

    if place_first:
        if step % 2 == 1:
            return True
    if step % 2 == 0:
        return True
    return False

def direction_to(src, target):
    """
    Returns the direction to a target
    """
    d_s = target - src
    d_x = d_s[0]
    d_y = d_s[1]
    if d_x == 0 and d_y == 0:
        return 0
    if abs(d_x) > abs(d_y):
        if d_x > 0:
            return 2
        return 4
    if d_y > 0:
        return 3
    return 1
