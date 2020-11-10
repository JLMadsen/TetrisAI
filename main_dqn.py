import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import sys
import copy
from datetime import datetime

import enviorment.util as util
from enviorment.colors import green, fail, header, cyan, warning
from enviorment.tetris import Tetris
from dqn.agent import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Tetris({'reduced_shapes':1})
agent = DQN(env)#.to(device)

load_weights = 0
plot = 0
train = 1
epoch = 30_000
epoch_time = 0
start_time = 0

def train():
    print(header('Train model: ')+cyan(str(epoch)))

    scores = []

    if load_weights:
        agent.load_weights('_new')
        
    agent.init_eps(epoch)
    
    for e in range(epoch):
        start_time = datetime.now()
        
        # print training info every 100th epoch
        if not e%(epoch//100): 
            print('\nTraining  : '+ str((progress := round(e/epoch*100, 2))) +' %')
            print('Highscore : ' + green(str(env.highscore)))
            if scores:
                print('    avg       :', (sum(scores)/len(scores)))
                if epoch_time:
                    print('epoch time: {}'.format(epoch_time*epoch//100))
                    print('eta       : {}'.format((epoch_time*(100-progress)*epoch//100)))
                    
                [print(s, end=', ') for s in [*map(lambda x: green(str(x)) if x == sorted(scores)[-1] else x, scores)]]
            print()
            
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
            
            if done:
                reward -= 100
            else:
                reward *= 100
            
            agent.memory.append([old_state, action, state, reward])
            
            epoch_time = datetime.now() - start_time
            
        # etter hver runde?
        if train:
            
            # hvor ofte?
            agent.cached_q_net = copy.deepcopy(agent.q_net)
            
            agent.epsilon -= agent.epsilon_decay
            
            agent.train_weights()
            
        if score:
            scores.append(score)
            
    print(scores)
    suffix = str(epoch//1000)+'k' if epoch>1000 else str(epoch)
    agent.save_weights('_'+suffix+'_2')
    
    if plot and scores:
        plt.plot([*range(len(scores))], scores)
        plt.show()

def run(weight=''):
    print(header('Run trained model'))
    scores = []
    agent.load_weights(weight)
    agent.epsilon = -1
    try:
        while 1:
            state, reward, done, info = env.reset()
            score = 0
            while not done:
                action = agent.policy(state)
                state, reward, done, info = env.step(action)
                print(warning(env.actionName(action)))
                if reward:
                    print(green('CLEARED LINE'))
                env.render()
                time.sleep(0.001)
            print(fail('RESET'))
            if score:
                scores.append(score)
    # break loop on CTRL C or quit pygame window
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
            
    if plot and scores:
        plt.plot([*range(len(scores))], scores)
        plt.show()

if __name__ == "__main__":
    try:
        #train() #7:00
        run('_60k')
        
    except KeyboardInterrupt:
        agent.save_weights('_quit')
    finally:
        env.quit()
        
