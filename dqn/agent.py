import math
import random
import numpy as np
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
        """
                
        self.model = nn.Sequential(
            nn.Conv2d(2, 1, (20, 10)),
            nn.ReLU(),
            nn.Conv2d(1, 1, (1, 5)),
            nn.ReLU(),
            nn.Linear(env.action_space, 1),
            nn.ReLU(),
        )
        
        #self.optimizer = optim.Adam(self.parameters, self.alpha)
        
    def forward(self, x):
        return self.model(x)
    
    def policy(self, state):
        if not torch.is_tensor(state):
            state = torch.Tensor([state])
        
        if random.uniform(0, 1) < self.epsilon and 0:
            return self.env.action_sample
        else:
            print((action := self.forward(state)))
            return np.argmax(action.detach().numpy())
    
    def save_weights(self, suffix=''):
        torch.save(self.state_dict(), weight_path+suffix)

    def load_weights(self):
        self.load_state_dict(torch.load(weight_path))
        self.eval()
    
    def train_weights(self, epochs=100):
        rewards = []
        scores = []
        steps = 0 # if we want to limit training
        
        for e in range(1):
            
            total_reward = 0
            obs, done, reward, info  = self.env.reset()
        
            possible_states = self.env.get_all_states()
            
            
        
            #while not done:
                
                #action, state,         
    
    def adjust_weights(self, epochs=100, batch_size=512):
        
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
               
        batch = self.memory.sample(batch_size)
        
        train_x = []
        train_y = []
    
    