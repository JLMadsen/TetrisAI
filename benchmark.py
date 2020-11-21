import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
import time
import torch

from enviorment.tetris import Tetris

from Imitation.agent import imitation_agent

from nat_selection.model import Model

from dqn.agent import DQN

plt_colors = ['r', 'g', 'b', 'm', "c"]
plt_light_colors = ['pink', 'palegreen', 'powderblue', 'thistle', "skyblue" ]

class randomAgent:
    def __init__(self, env):
        self.env = env
        self.name = 'Random'
    def policy(self, *args):
        return self.env.action_sample

def main():
    env = Tetris({'reduced_shapes': 1})

    
    agent1 = imitation_agent(env)
    agent1.load_weights('_10k_0,1_nat1')
    agent1.epsilon = -1
    agent1.name = '10k epoker 2k tupler 0.1 læringsrate'

    agent2 = imitation_agent(env)
    agent2.load_weights('_10k_0.01_nat1')
    agent2.epsilon = -1
    agent2.name = '10k epoker 2k tupler 0.01 læringsrate'


    agent3 = imitation_agent(env)
    agent3.load_weights('_10k_0,1_nat2')
    agent3.epsilon = -1
    agent3.name = '10k epoker 20k tupler 0.1 læringsrate'


    agent4 = imitation_agent(env)
    agent4.load_weights('_10k_0,01_nat2')
    agent4.epsilon = -1
    agent4.name = '10k epoker 20k tupler 0.01 læringsrate'


        
    agents = [agent1, agent2, agent3, agent4]
    agent_labels = [a.name for a in agents]
    agent_scores = {}
    
    sample = 20
    
    agents = [deepcopy(agents) for _ in range(sample)]
    agents = [a for n in agents for a in n]
    
    for agent in agents:
        current_agent = agent.name
        
        max_actions = 2000
        actions = 0
        #random.seed(420)
        #np.random.seed(420)
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
                
                if isinstance(agent, Model):
                    action = agent.best(env)
                elif isinstance(agent, imitation_agent):
                    action = agent.f(state if torch.is_tensor(state) else torch.tensor(state).unsqueeze(0).float()).argmax(1)
                else:
                    action = agent.policy(state)
                    
                if isinstance(action, list):
                    for a in action:
                        state, reward, done, info = env.step(a)
                        agent_scores[current_agent][-1].append(reward)
                        actions+=1
                else:
                    state, reward, done, info = env.step(action)
                    agent_scores[current_agent][-1].append(reward)
                    actions+=1

    for agent, scores in agent_scores.items():

        index = agent_labels.index(agent)
        
        scores = [*map(lambda x: np.cumsum(x).tolist(), scores)]

        avg = [sum(s)/len(s) for s in [*zip(*scores)]]
        
        #for score in scores:
            #plt.plot([*range(len(score))], score, c=plt_light_colors[index])
            
        plt.plot([*range(len(avg))], avg, c=plt_colors[index], label=agent_labels[index])
        
    plt.ylabel('Poengsum')
    plt.xlabel('Handlinger')    
    
    uuid = str(time.time()).split(".")[0][-5:]
    
    plt.legend()
    plt.text(0.15, .94, 'Spill = '+ str(sample), fontsize=12, transform=plt.gcf().transFigure)
    plt.savefig('./rapporter/imgs/comparison_'+uuid+'.png')
    plt.show()
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
