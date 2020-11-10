def _getHeights(state):
    height = [len(state) for _ in range(len(state[0]))]
    for j in range(len(state[0])):
        for i in range(len(state)):
            if state[i][j] == 1:
                height[j] = i
                break
    return [abs(i - len(state)) for i in height]


def holes(state):
    holes = 0
    for j in range(len(state[0])):
        checkingHole = False
        for i in range(len(state)):
            if state[-(i+1)][j] == 0:
                checkingHole = True
            elif state[-(i+1)][j] == 1:
                if checkingHole:
                    checkingHole = False
                    holes += 1
    return holes


def totalHeight(state):
    return sum(_getHeights(state))


def maxHeight(state):
    return max(_getHeights(state))


def evenness(state):
    heights = _getHeights(state)
    return sum([i for i in [abs(heights[j-1]-heights[j]) for j in range(1, len(heights))]])