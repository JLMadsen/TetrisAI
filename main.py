"""
Tetris in Python for DQN
Run this for manual testing of Tetris Enviorment
"""

from enviorment.tetris import Tetris

env = Tetris({'reduced_shapes': 0})


def main():
    while 1:
        env.reset()
        done = False
        while not done:
            state, action, done = env.render(1)


if __name__ == "__main__":
    main()
