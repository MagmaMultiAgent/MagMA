"""Module containing utility functions"""
def my_turn_to_place_factory(place_first: bool, step: int):
    """Function returning if it is player's turn to place factory"""
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    """Function returning direction number for target direction"""
    dir_s = target - src
    dir_x = dir_s[0]
    dir_y = dir_s[1]
    if dir_x == 0 and dir_y == 0:
        return 0
    if abs(dir_x) > abs(dir_y):
        if dir_x > 0:
            return 2
        return 4
    else:
        if dir_y > 0:
            return 3
        return 1
