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

        # learning rate
        self.alpha = .001
        
        # discount
        self.gamma = .9
        
        # exploration rate
        self.upper_epsilon = 1
        self.lower_epsilon = .01
        self.epsilon_decay = 0
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
                
        self.q_net = nn.Sequential(
            nn.Conv2d(2, 32, (20, 10)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (1, 1)),
            nn.ReLU(),
            nn.Linear(1, env.action_space)
        )
        
        self.cached_q_net = deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), self.alpha)
        self.loss = nn.MSELoss()
        self.loss_temp = 0
        self.loss_count = 0
        
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.Tensor([x])
            
        return self.q_net(x)
    
    def brute(self, state):
        # Check all states and choose max reward
        states, actions, rewards = self.env.get_all_states(state)
        return actions[np.argmax(rewards)]
    
    def init_eps(self, epochs):
        self.epsilon_decay = (self.upper_epsilon - self.lower_epsilon) / epochs
    
    def policy(self, state):
        
        # https://docs.scipy.org/doc//numpy-1.10.4/reference/generated/numpy.random.choice.html
        
        if random.uniform(0, 1) < self.epsilon:
            
            return self.env.action_sample
        else:
            
            actions = self.forward(state)
            actions = actions.detach().numpy()
            actions = actions[0][0][0]
            return np.argmax(actions)
    
    def save_weights(self, suffix=''):
        torch.save(self.state_dict(), weight_path+suffix)
    
    def ex(self, tens):
        return tens[0][0][0]
        
    def load_weights(self, suffix=''):
        self.load_state_dict(torch.load(weight_path+suffix))
        self.eval()
    
    # https://github.com/CogitoNTNU/vicero/blob/678f4f139788cb9be149f6d9651d93ca737aeccd/vicero/algorithms/deepqlearning.py#L140
    def train_weights(self, batch_size=100):
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
                    
        if not batch_size:
            return

        batch = self.memory.sample(batch_size)
        
        for state, action, next_state, reward in batch:
            state = torch.tensor([state]).float()
            next_state = torch.tensor([next_state]).float()
            reward = torch.tensor(reward).float()
            
            target = reward
            
            outputs = self.ex(self.cached_q_net(next_state))
            target = (reward + self.gamma * torch.max(outputs))
            
            target_f = self.ex(self.q_net(state))
            target_f[action] = target
            
            prediction = self.ex(self.q_net(state))
            loss = self.loss(prediction, target_f)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()