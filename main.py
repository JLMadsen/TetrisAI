"""
Tetris in Python for DQN

Run this for manual testing of Tetris Enviorment
"""
from nat_selection.agent import Agent as NatAgent
from nat_selection.model import Model
import time
from enviorment.tetris import Tetris

env = Tetris({'reduced_shapes': 0})


def main():

    agent = NatAgent(cores=4)

    epoch = 100_000

    candidate = agent.train(epoch)
    #candidate = Model(-0.8995652940240592, 0.06425443268253492, -0.3175211096545741, -0.292974392382306)

    while True:
        score = 0
        state, reward, done, info = env.reset()

        while not done:

            action = candidate.best(env)

            for a in action:
                state, reward, done, info = env.step(a)
                score += reward

            env.render()
            time.sleep(0)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        pass

    finally:
        env.quit()

"""

Spørsmål:

Hva skal NN ta inn?
    
    State + action
    
    heuristic values

Gi reward for brikke plassering?

Sjekke enkelte action eller "slutt states"?

    se på alle states, fortere belønning, mindre å sjekke med NN


"""
