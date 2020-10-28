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
        
        self.alpha = .001
        self.gamma = .9
        self.upper_epsilon = 1
        self.lower_epsilon = .01
        
        self.memory = Memory()
        
        # replace with correct layers
        self.model = nn.Sequential(
            nn.conv3d(),
            nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            #nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.Linear(self.feature_size(), feature_dimension),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.model(x)
    
    def train(self):
        pass
    