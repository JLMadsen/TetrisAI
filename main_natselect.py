"""
Tetris in Python for Natural Selection
"""
from nat_selection.agent import Agent as NatAgent
from nat_selection.model import Model
import time
from enviorment.tetris import Tetris

env = Tetris({'reduced_grid': 0, 'reduced_shapes': 0})


def main():
    agent = NatAgent(cores=4)

    generations = 1000

    #candidate = agent.train(generations)
    candidate = Model([-0.8995652940240592, 0.06425443268253492, -0.3175211096545741, -0.292974392382306])

    while True:
        score = 0
        state, reward, done, info = env.reset()

        while not done:

            action = candidate.best(env)

            for a in action:
                env.render()
                time.sleep(0.1)
                state, reward, done, info = env.step(a)
                score += reward

            


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        pass

    finally:
        env.quit()
