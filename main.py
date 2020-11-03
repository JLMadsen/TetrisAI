from enviorment.tetris import Tetris

from dqn.agent import DQN

import numpy as np
import time
import matplotlib.pyplot as plt

env = Tetris({'reduced_shapes': 0})
agent = DQN(env)

def actionName(action):
    attrs = [a for a in dir(env.actions) if not a.startswith('__')]
    for attr in attrs:
        value = env.actions.__getattribute__(env.actions, attr)
        if isinstance(value, int) and value == action:
            return attr


manual = 0
load_weights = 0
plot = 1
train = 1

def main():

    if manual:
        while 1:
            env.reset()
            done = False
            while not done:
                state, action, done = env.render(1)
                
                
    else:
        scores = []
        epoch = 100

        #if load_weights:
        #    agent.load_weights()
        #else:
        #    agent.train_weights()
        
        for e in range(epoch):
            
            if not e%10: print('Epoch:', e)
            
            score = 0
            state, reward, done, info = env.reset()
            
            while not done:
                
                action = agent.policy(state) 
                
                if isinstance(action, list):
                    for a in action:
                        state, reward, done, info = env.step(a)
                        score += reward
                else:
                    print('main: action:', actionName(action), action)
                    state, reward, done, info = env.step(action)
                    score += reward

                #                                            next_state
                experience = agent.Transition(state, action, None, reward)
                agent.memory.append(experience)

                env.render()
                time.sleep(0.07 if e < 0 else 0)
                
                if train: agent.train_weights()
                
            if score != 0:
                scores.append(score)
                
        print(scores)
        #agent.save_weights('_new')
        
        if plot and scores:
            plt.plot(list(range(len(scores))), scores)
            plt.show()

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
