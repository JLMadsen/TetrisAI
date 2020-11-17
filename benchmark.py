import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy

from enviorment.tetris import Tetris

from nat_selection.model import Model

from dqn.agent import DQN

plt_colors = ['r', 'g', 'b']
plt_light_colors = ['pink', 'palegreen', 'powderblue']

class randomAgent:
    def __init__(self, env):
        self.env = env
    def policy(self, *args):
        return self.env.action_sample

def main():
    env = Tetris({'reduced_shapes': 1, 'reduced_grid': 1})

    agent1 = DQN(env)
    agent1.load_weights('_60k_2')
    agent1.epsilon = -1
    
    agent2 = Model([-0.8995652940240592, 0.06425443268253492, -0.3175211096545741, -0.292974392382306])

    agent3 = randomAgent(env)

    agent_labels = ['Imitation + DQN', 'Natural Selection', 'Random']
    agents = [agent1, agent2, agent3]
    agent_scores = {}
    sample = 5
    
    agents = [deepcopy(agents) for _ in range(sample)]
    agents = [a for n in agents for a in n]
    
    """for a in agents:
        print(a.__class__.__name__)
    exit()"""
    
    for agent in agents:
        current_agent = agent.__class__.__name__
        
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

    agent_colors = sorted(agent_scores.keys())
    for agent, scores in agent_scores.items():
        index = agent_colors.index(agent)
        
        scores = [*map(lambda x: np.cumsum(x).tolist(), scores)]

        avg = [sum(s)/len(s) for s in [*zip(*scores)]]
        
        for score in scores:
            plt.plot([*range(len(score))], score, c=plt_light_colors[index])

        plt.plot([*range(len(avg))], avg, c=plt_colors[index], label=agent_labels[index])
        
    plt.ylabel('Score')
    plt.xlabel('Actions')    
    
    plt.legend()
    plt.savefig('./rapporter/imgs/comparison.png')
    plt.show()
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
