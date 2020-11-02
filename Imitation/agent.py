import torch
import torch.nn as nn
import torchvision
from pathlib import Path
mod_path = Path(__file__).parent
weight_path = str(mod_path) + '/weights'

class imitation_agent(nn.Module):
    def __init__(self, env):
        super(imitation_agent, self).__init__()

        self.env = env

        # Model layers (includes initialized model variables):
        self.conv = nn.Conv2d(2, 32, (20, 10))
        self.conv2 = nn.Conv2d(32, 64, (1, 1))
        self.dense = nn.Linear(64 * 1 * 1, env.action_space)
        self.ReLU = nn.ReLU()
        

    def logits(self, x):

        x = self.conv(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.dense(x.reshape(-1, 64*1*1))
        x = self.ReLU(x)
        return x

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


    def save_weights(self, suffix=''):
        torch.save(self.state_dict(), weight_path+suffix)

    def load_weights(self):
        self.load_state_dict(torch.load(weight_path))
        self.eval()