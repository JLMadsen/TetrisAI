import math
import random
import numpy as np
from copy import deepcopy

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
        
        self.model = nn.Sequential(
            nn.conv3d(10*20*2, ),
            nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            #nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Linear(env.action_space, 1),
            nn.ReLU(),
        )
        
        self.optimizer = optim.Adam(self.parameters, self.alpha)
        
    def forward(self, x):
        return self.model(x)
    
    def policy(self):
        pass
    
    