import enviorment.util as util
from enviorment.tetris import Tetris
import numpy as np
import copy


class Model():
    def __init__(self, heightWeight, clearedWeight, holesWeight, evennessWeight):
        self.heightWeight = heightWeight
        self.clearedWeight = clearedWeight
        self.holesWeight = holesWeight
        self.evennessWeight = evennessWeight
        self.fitness = 0

    def getWeights(self):
        return self.heightWeight, self.clearedWeight, self.holesWeight, self. evennessWeight

    def best(self, env):
        checkpoint = env.save_checkpoint()

        states, actions, rewards = env.get_all_states()

        scores = []
        for i, state in enumerate(states):
            for _, action in enumerate(actions[i]):
                env.step(action)

            scores.append(self.heightWeight * util.totalHeight(state) + self.clearedWeight *
                          rewards[i] + self.holesWeight * util.holes(state) + self.evennessWeight * util.evenness(state))

            env.load_checkpoint(checkpoint)

        return actions[np.argmax(scores)]
