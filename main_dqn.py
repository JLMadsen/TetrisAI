from enviorment.tetris import Tetris

from dqn.agent import DQN

import numpy as np
import time
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Tetris({'reduced_shapes': 1})
agent = DQN(env)#.to(device)

def actionName(action):
    attrs = [a for a in dir(env.actions) if not a.startswith('__')]
    for attr in attrs:
        value = env.actions.__getattribute__(env.actions, attr)
        if isinstance(value, int) and value == action:
            return attr

load_weights = 0
plot = 1
train = 1

def main():

    scores = []
    epoch = 100

    if load_weights:
        agent.load_weights()
    
    for e in range(epoch):
        
        if not e%10: print('Epoch:', e)
        
        score = 0
        state, reward, done, info = env.reset()
        
        while not done:
            old_state = state
            
            action = agent.policy(state) 
            
            if isinstance(action, list):
                for a in action:
                    state, reward, done, info = env.step(a)
                    score += reward
            else:
                print('main: action:', actionName(action), action)
                state, reward, done, info = env.step(action)
                score += reward

            #experience = agent.Transition(old_state, action, state, reward)
            #agent.memory.append(experience)
            
            agent.memory.append([old_state, action, state, reward])

            env.render()
            time.sleep(0.07 if e < 0 else 0)
            
        if train: agent.train_weights()
            
        if score != 0:
            scores.append(score)
            
    print(scores)
    agent.save_weights('_new')
    
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