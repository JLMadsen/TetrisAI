import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import sys
import copy

from enviorment.tetris import Tetris
from dqn.agent import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Tetris({'reduced_shapes':1})
agent = DQN(env)#.to(device)

def actionName(action):
    attrs = [a for a in dir(env.actions) if not a.startswith('__')]
    for attr in attrs:
        value = env.actions.__getattribute__(env.actions, attr)
        if isinstance(value, int) and value == action:
            return attr

load_weights = 0
plot = 0
train = 1

def main():

    scores = []
    epoch = 10_000

    if load_weights:
        agent.load_weights('_new')
        
    agent.init_eps(epoch)
    
    for e in range(epoch):
        
        if not epoch//100 or not e%(epoch//100): 
            print('\nTraining: '+ str(round(e/epoch*100, 2)) +' %')
            print('Highscore : '+ str(env.highscore))
        
        score = 0
        action = 0
        time_alive = 0
        state, reward, done, info = env.reset()
        
        while not done:
            old_state = state
            time_alive += 1
            
            action = agent.policy(state)
            state, reward, done, info = env.step(action)
            score += reward
            
            reward = (reward*100)+sum(env.heuristic_value(state))
            #print(reward)
            agent.memory.append([old_state, action, state, reward])

            if e > epoch - 10:
                print('Action:', actionName(action))
                env.render()
                time.sleep(0.05)
            
        if train:
            if not e%(epoch/10):
                agent.cached_q_net = copy.deepcopy(agent.q_net)
            
            agent.epsilon -= agent.epsilon_decay
            agent.train_weights()
            
        if score:
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
        agent.save_weights('_quit')
    
    finally:
        env.quit()