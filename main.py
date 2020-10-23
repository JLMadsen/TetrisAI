from enviorment.tetris import Tetris
from enviorment.actions import Action

import numpy as np

# Manual testing of tetris env

env = Tetris()

def main(manual=0):

    if manual:
        while 1:
            env.reset()
            done = False
            while not done:
                done = env.render(1)
    else:
        while 1:
            state, reward, done, info = env.reset()
            
            while not done:
                
                action = np.random.randint(len(Action.ALL))
                state, reward, done, info = env.step(action)
                env.render()

if __name__ == "__main__":
    main()
