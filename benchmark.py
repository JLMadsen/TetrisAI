import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
from copy import deepcopy

from enviorment.tetris import Tetris

from Imitation.agent import imitation_agent
from nat_selection.model import Model
from dqn.agent import DQN

# Colors for plotting
plt_colors = ['r', 'g', 'b', 'm', 'c']
plt_light_colors = ['pink', 'palegreen', 'powderblue', 'thistle', 'lightcyan']

# Random agent to show difference between trained models and random.
class randomAgent:
    def __init__(self, env):
        self.env = env
        self.name = 'Tilfeldig'
    def policy(self, *args):
        return self.env.action_sample

def main():
    env = Tetris({'reduced_shapes': 1, 'reduced_grid': 0})
    
    agent1 = DQN(env)
    agent1.load_weights('_60k_3')
    agent1.epsilon = 0
    agent1.name = 'DQN'
    
    agent2 = imitation_agent(env)
    agent2.load_weights('_10k_01_nat1')
    agent2.name = 'Imitasjon'
    
    agent3 = DQN(env)
    agent3.load_weights('_60k_imitation_3')
    agent3.epsilon = 0
    agent3.name = 'Imitasjon + DQN'
        
    agent4 = Model([-0.8995652940240592, 0.06425443268253492, -0.3175211096545741, -0.292974392382306])
    
    agent5 = randomAgent(env)
    
    # which agents to test.
    agents = [agent1, agent2, agent5]
    agent_labels = [a.name for a in agents]
    agent_scores = {}
    
    # how samples of each agent
    sample = 30
    
    agents = [deepcopy(agents) for _ in range(sample)]

    # flatten list
    agents = [a for n in agents for a in n]
    
    for agent in agents:
        current_agent = agent.name
        print('Sampling', current_agent.ljust(10), '| Progress', str(agents.index(agent)).rjust(2), '/', len(agents))
        
        max_actions = 1000
        actions = 0

        if current_agent in agent_scores.keys():
            agent_scores[current_agent].append([])
        else:
            agent_scores[current_agent] = [[]]
        
        while 1:
            
            state, reward, done, info = env.reset()
            
            if actions >= max_actions:
                break
            
            while not done:
                
                if actions >= max_actions:
                    break
                
                # Get action from agent spesific method.
                if isinstance(agent, Model):
                    action = agent.best(env)
                elif isinstance(agent, imitation_agent):
                    action = agent.f(state if torch.is_tensor(state) else torch.tensor(state).unsqueeze(0).float()).argmax(1)
                else:
                    action = agent.policy(state)
                    
                # If agent returns list of actions to perform.
                if isinstance(action, list):
                    for a in action:
                        state, reward, done, info = env.step(a)
                        agent_scores[current_agent][-1].append(reward)
                        actions+=1
                else:
                    state, reward, done, info = env.step(action)
                    agent_scores[current_agent][-1].append(reward)
                    actions+=1

    # Plot scores.    
    for agent, scores in agent_scores.items():
        index = agent_labels.index(agent)
        scores = [*map(lambda x: np.cumsum(x).tolist(), scores)]

        avg = [sum(s)/len(s) for s in [*zip(*scores)]]
        
        for score in scores:
            plt.plot([*range(len(score))], score, c=plt_light_colors[index])
            
        plt.plot([*range(len(avg))], avg, c=plt_colors[index], label=agent_labels[index])
        
    plt.ylabel('Poengsum')
    plt.xlabel('Handlinger')    
    
    plt.legend()
    plt.text(0.15, .9, 'Spill = '+ str(sample), fontsize=12, transform=plt.gcf().transFigure)
    
    # uncomment to save image
    #plt.savefig('./rapporter/imgs/comparison_'+str(time.time()).split(".")[0][-5:]+'.png')
    plt.show()
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
