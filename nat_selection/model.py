import enviorment.util as util
from enviorment.tetris import Tetris
import numpy as np
import copy


weightsNum = len(util.heuristics)


class Model():
    def __init__(self, weights):
        self.weights = weights
        self.fitness = 0
        self.name = 'Genetisk algoritme'

    def best(self, env):
        checkpoint = env.save_checkpoint()

        states, actions, rewards = env.get_all_states()

        scores = []

        for i, state in enumerate(states):
            for _, action in enumerate(actions[i]):
                env.step(action)

            util.heuristics[util.clearedLinesIndex] = rewards[i]

            score = 0
            for h, w in zip(util.heuristics, self.weights):
                score += (
                    float(h(state))*w if hasattr(h, '__call__') else float(h)*w)

            scores.append(score)
            env.load_checkpoint(checkpoint)

        return actions[np.argmax(scores)]
