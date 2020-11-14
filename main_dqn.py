import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import sys
import copy

import enviorment.util as util
from enviorment.colors import green, fail, header, cyan, warning
from enviorment.tetris import Tetris
from dqn.agent import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Tetris({
     'reduced_shapes': 1
    ,'reduced_grid': 1
})

agent = DQN(env)#.to(device)

def train(plot=0, epoch=60_000):
    print(header('Train model: ')+cyan(str(epoch)))

    scores = []
        
    agent.init_eps(epoch)
    
    for e in range(epoch):
        
        # print training info every 100th epoch
        if not e%(epoch//100): 
            print('\nTraining  : '+ str((progress := round(e/epoch*100, 0))) +' %')
            print('Highscore : ' + green(str(env.highscore)))
            if scores:
                print('avg       :', round(sum(scores)/len(scores), 2))
                [print(s, end=', ') for s in [*map(lambda x: green(str(x)) if x == sorted(scores)[-1] else x, scores)]]
            print()
            
        score = 0
        action = 0
        time_alive = 0
        state, reward, done, info = env.reset()
        
        while not done:
            old_state = copy.deepcopy(state)
            time_alive += 1
            
            action = agent.policy(state)
            state, reward, done, info = env.step(action)
            score += reward
            
            if done:
                reward = -1
            
            agent.memory.append([old_state, action, state, reward])
                        
        if not e%1000:
            agent.cached_q_net = copy.deepcopy(agent.q_net)
                
        agent.train_weights(30)
        
        agent.epsilon -= agent.epsilon_decay
            
        if score:
            scores.append(score)
            
    print(scores)
    suffix = str(epoch//1000)+'k' if epoch>1000 else str(epoch)
    agent.save_weights('_'+suffix+'_3')
    
    if plot and scores:
        plt.plot([*range(len(scores))], scores)
        plt.show()

def run(weight='', attempts=300):
    print(header('Run trained model'))
    scores = []
    agent.load_weights(weight)
    agent.epsilon = -1
    try:
        while 1:
            if attempts == env.attempt:
                break
            
            state, reward, done, info = env.reset()
            score = 0
            while not done:
                action = agent.policy(state)
                state, reward, done, info = env.step(action)
                #print(warning(env.actionName(action)))
                if reward:
                    score += reward
                    print(green('CLEARED LINE'))
                #env.render()
                #time.sleep(0.001)
            print(fail('RESET'))
            scores.append(score)
    # break loop on CTRL C or quit pygame window
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
            
    if scores:
        plt.plot([*range(len(scores))], scores, label='score')
        plt.legend()
        plt.text(0.2, .94, 'Average score = '+ str(round(sum(scores)/len(scores), 2)), fontsize=12, transform=plt.gcf().transFigure)
        plt.text(0.6, .94, 'weight = '+ weight, fontsize=12, transform=plt.gcf().transFigure)
        plt.show()

if __name__ == "__main__":
    plot = 0
    epoch = 100_000
    
    try:
        
        #agent.load_weights('_60k_2')
        #agent.upper_epsilon = agent.epsilon = .5
        #train(plot, epoch) # 7:23
        
        run('_60k_2')
        
    except KeyboardInterrupt:
        agent.save_weights('_quit')
    finally:
        env.quit()
        

