import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy

from enviorment.tetris import Tetris

from nat_selection.model import Model

from dqn.agent import DQN

plt_colors = ['r', 'g', 'b', 'm', 'c']
plt_light_colors = ['pink', 'palegreen', 'powderblue', 'thistle', 'lightcyan']

class randomAgent:
    def __init__(self, env):
        self.env = env
        self.name = 'Random'
    def policy(self, *args):
        return self.env.action_sample

def main():
    env = Tetris({'reduced_shapes': 1, 'reduced_grid': 1})
    
    agent1 = DQN(env)
    agent1.load_weights('_60k_2')
    agent1.epsilon = -1
    agent1.name = 'DQN'
    
    agent2 = imitation_agent(env)
    agent2.load_weights('_10k_01_nat1')
    agent2.name = 'Imitation'
    
    agent2 = DQN(env)
    agent2.load_weights('_60k_imitation')
    agent2.epsilon = -1
    agent2.name = 'Imitation + DQN'
        
    agent4 = Model([-0.8995652940240592, 0.06425443268253492, -0.3175211096545741, -0.292974392382306])
    
    agent5 = randomAgent(env)
        
    agents = [agent1, agent2,  agent3, agent5]#, agent5]
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
        
        for score in scores:
            #plt.plot([*range(len(score))], score, c=plt_light_colors[index])
            a=1
            
        plt.plot([*range(len(avg))], avg, c=plt_colors[index], label=agent_labels[index])
        
    plt.ylabel('Poengsum')
    plt.xlabel('Handlinger')    
    
    plt.legend()
    plt.text(0.15, .5, 'Spill = '+ str(sample), fontsize=12, transform=plt.gcf().transFigure)
    plt.savefig('./rapporter/imgs/comparison_'+uuid+'.png')
    plt.show()
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
