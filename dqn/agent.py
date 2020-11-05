import math
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
mod_path = Path(__file__).parent
weight_path = str(mod_path) + '/weights'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from dqn.memory import Memory

"""
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://jonathan-hui.medium.com/rl-dqn-deep-q-network-e207751f7ae4
https://drive.google.com/file/d/0BxXI_RttTZAhVUhpbDhiSUFFNjg/view
"""

class DQN(nn.Module):
    
    def __init__(self, env):
        super(DQN, self).__init__()
        
        self.env = env
        self.Transition = namedtuple(
                        'Transition',
                        ('state', 'action', 'next_state', 'reward'))

        # learning rate
        self.alpha = .001
        
        # discount
        self.gamma = .9
        
        # exploration rate
        self.upper_epsilon = 1
        self.lower_epsilon = .01
        self.epsilon = self.upper_epsilon
        
        self.memory = Memory()
                
        """
        Conv2d 1:
            in_channels
                2, grid_layer + piece_layer
            out_channels
                ~ TBD
            kernel_size
                (20, 10) = (height * width)
        
        Conv2d 2:
            in_channels
                32
            out_channels
                64
            kernel_size
                (1, 1)
        """
                
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, (20, 10)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (1, 1)),
            nn.ReLU(),
            nn.Linear(1, env.action_space)
        )
                
        self.optimizer = optim.Adam(self.model.parameters(), self.alpha)
        #self.loss = F.smooth_l1_loss
        
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.Tensor([x])
            
        return self.model(x)
    
    def policy(self, state):
        
        # https://docs.scipy.org/doc//numpy-1.10.4/reference/generated/numpy.random.choice.html
        
        if random.uniform(0, 1) < self.epsilon and 0:
            return self.env.action_sample
        else:
            
            # Check all states and choose max reward
            #states, actions, rewards = self.env.get_all_states(state)                
            #return actions[np.argmax(rewards)]

            actions = self.forward(state)

            actions = actions.detach().numpy()

            actions = actions[0][0][0]

            print(actions)

            return np.argmax(actions)
                
    
    def save_weights(self, suffix=''):
        torch.save(self.state_dict(), weight_path+suffix)

    def load_weights(self, suffix=''):
        self.load_state_dict(torch.load(weight_path+suffix))
        self.eval()
            
    def train_weights(self, batch_size=512):
        
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
                    
        if not batch_size:
            return

        batch = self.memory.sample(batch_size)
        #batch = [self.Transition(*zip(*b)) for b in batch]

        #non_final_mask
        #non_final_mask_states
        
        return
        
        print(batch)
        
        state_batch  = torch.cat([torch.tensor(s[0]) for s in batch])
        action_batch = torch.cat([torch.tensor(s[1]) for s in batch])
        reward_batch = torch.cat([torch.tensor(s[2]) for s in batch])
        
        print(action_batch)
        
        #state_batch  = torch.cat(  [torch.tensor(s) for s in batch[0] ]  )
        #action_batch = torch.cat(  [torch.tensor(a) for a in batch.action]  )
        #reward_batch = torch.cat(  [torch.tensor(r) for r in batch.reward]  )
        
        
        # do stuff
        
        
        self.optimizer.zero_grad()
        #self.loss.backward()
        self.optimizer.step()
        
        
        
        
        
        
        
        
        