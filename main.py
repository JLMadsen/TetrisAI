from enviorment.tetris import Tetris

from dqn.agent import DQN

import numpy as np
import time
import matplotlib.pyplot as plt

env = Tetris({'reduced_shapes': 0})
model = DQN(env)

def actionName(action):
    attrs = [a for a in dir(env.actions) if not a.startswith('__')]
    for attr in attrs:
        value = env.actions.__getattribute__(env.actions, attr)
        if isinstance(value, int) and value == action:
            return attr

def main(manual=0, load_weights=False, plot=True):

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
        #    model.load_weights()
        #else:
        #    model.train_weights()
        
        for e in range(epoch):
            
            if not e%10:
                print('Epoch:', e)
            
            score = 0
            state, reward, done, info = env.reset()
            
            while not done:
                
                action = model.policy(state) 
                
                               
                if isinstance(action, list):
                    for a in action:
                        state, reward, done, info = env.step(a)
                        score += reward
                else:
                    print('main: action:', actionName(action), action)
                    state, reward, done, info = env.step(action)
                    score += reward


                env.render()
                time.sleep(0.07 if e < 0 else 0)
                
            if score != 0:
                scores.append(score)
                
        print(scores)
        #model.save_weights('_new')
        
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
