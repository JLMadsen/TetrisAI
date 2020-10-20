class Action:

    WAIT   = 0 # no move, just one tile down
    ROTATE = 1
    LEFT   = 2
    RIGHT  = 3
    DOWN   = 4

    ALL = [WAIT, ROTATE, LEFT, RIGHT, DOWN]