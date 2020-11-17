import matplotlib.pyplot as plt
import numpy as np

from enviorment.colors import green, fail, header, cyan, warning
from enviorment.tetris import Tetris

from nat_selection.agent import Agent as NatAgent
from nat_selection.model import Model

from dqn.agent import DQN

def main():
    env = Tetris({'reduced_shapes': 1, 'reduced_grid': 1})

    agent1 = DQN(env)
    agent1.load_weights('_60k_2')
    agent1.epsilon = -1
    
    agent2 = Model([-0.8995652940240592, 0.06425443268253492, -0.3175211096545741, -0.292974392382306])

    agents = [agent1, agent2]
    agent_scores = []
    
    for agent in agents:
        
        max_actions = 500
        actions = 0
        agent_scores.append([])
        
        while 1:
            
            state, reward, done, info = env.reset()
            
            if actions >= max_actions:
                break
            
            while not done:
                
                if actions >= max_actions:
                    break
                
                if isinstance(agent, DQN):
                    action = agent.policy(state)
                else:
                    action = agent.best(env)
                    
                if isinstance(action, list):
                    for a in action:
                        state, reward, done, info = env.step(a)
                        agent_scores[-1].append(reward)
                        actions+=1
                else:
                    state, reward, done, info = env.step(action)
                    agent_scores[-1].append(reward)
                    actions+=1
                
    for agent, scores in zip(agents, agent_scores):
                
        cumsum = np.cumsum(scores).tolist()
        
        plt.plot([*range(len(scores))], cumsum, label=agent.__class__.__name__)
        
    plt.legend()
    plt.savefig('test.png')
    plt.show()
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
