from enviorment.tetris import Tetris

from dqn.agent import DQN

import numpy as np
import time

env = Tetris()
agent = DQN(env)

def main(manual=0):

    if manual:
        while 1:
            env.reset()
            done = False
            while not done:
                done = env.render(1)
    else:
        scores = []
        epoch = 100_000
        
        for e in range(epoch):
            
            if not e%500:
                print(e)
            
            score = 0
            state, reward, done, info = env.reset()
            
            while not done:
                
                action = env.action_sample             
                state, reward, done, info = env.step(action)
                                
                env.render()
                time.sleep(0.1 if e < 2 else 0)
                
                score += reward
                
            if score != 0:
                scores.append(score)
                
        print(scores)

if __name__ == "__main__":
    main()
