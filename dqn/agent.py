import math
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
mod_path = Path(__file__).parent
weight_path = str(mod_path) + '/weights/weights'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from dqn.memory import Memory
from dqn.modules import Resize, Print_shape

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
        self.gamma = .95
        
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
            kernel_size
                (20, 10) = (height * width)
        """
                
        self.q_net = nn.Sequential(
            nn.Conv2d(2, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Resize(-1, 192),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space)
            #,Print_shape('out')
        )
        
        self.cached_q_net = deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), self.alpha)
        self.loss = nn.MSELoss()
        self.loss_temp = 0
        self.loss_count = 0
            
    def brute(self, state):
        # Check all states and choose max reward
        states, actions, rewards = self.env.get_all_states(state)
        return actions[np.argmax(rewards)]
    
    def init_eps(self, epochs):
        self.epsilon_decay = (self.upper_epsilon - self.lower_epsilon) / epochs
    
    def policy(self, x):
        
        # https://docs.scipy.org/doc//numpy-1.10.4/reference/generated/numpy.random.choice.html
        
        if random.uniform(0, 1) < self.epsilon:
            
            return self.env.action_sample
        
        else:
            
            if not torch.is_tensor(x):
                x = torch.Tensor([x])
                                    
            actions = self.q_net(x).argmax()
            return actions            
    
    def save_weights(self, suffix=''):
        torch.save(self.q_net.state_dict(), weight_path+suffix)
    
    def load_weights(self, suffix=''):
        self.q_net.load_state_dict(torch.load(weight_path+suffix))
        self.eval()
    
    # https://github.com/CogitoNTNU/vicero/blob/678f4f139788cb9be149f6d9651d93ca737aeccd/vicero/algorithms/deepqlearning.py#L140
    def train_weights(self, batch_size=80):
        
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
                    
        if not batch_size:
            return

        batch = self.memory.sample(batch_size)
        
        for state, action, next_state, reward in batch:
            state = torch.tensor([state]).float()
            next_state = torch.tensor([next_state]).float()
            reward = torch.tensor(reward).float()
                        
            outputs = self.cached_q_net(next_state)
            
            target_f = self.q_net(state)
            target_f[0][action] = (reward + self.gamma * torch.max(outputs))
            
            prediction = self.q_net(state)
            loss = self.loss(prediction, target_f)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
