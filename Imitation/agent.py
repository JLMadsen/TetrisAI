import torch
import torch.nn as nn
import torchvision
from pathlib import Path
mod_path = Path(__file__).parent
weight_path = str(mod_path) + '/weights/weights'
from dqn.modules import Resize, Print_shape

class imitation_agent(nn.Module):
    def __init__(self, env):
        super(imitation_agent, self).__init__()
        self.name = 'Imitation'

        self.env = env

        dense_shape = resize_to = 64 if env.config['reduced_grid'] else 192

        self.q_net = nn.Sequential(
            nn.Conv2d(2, 32, 3),
            nn.LeakyReLU(.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(.1),
            nn.MaxPool2d(2, 2),
            Resize(-1, resize_to),
            nn.Linear(dense_shape, 64),
            nn.LeakyReLU(.1),
            nn.Linear(64, env.action_space),
            nn.LeakyReLU(.1)
        )

    # Predictor
    def f(self, x):
        return torch.softmax(self.q_net(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.q_net(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


    def save_weights(self, suffix=''):
        torch.save(self.q_net.state_dict(), weight_path+suffix)

    def load_weights(self, suffix=''):
        self.q_net.load_state_dict(torch.load(weight_path+suffix))
        self.eval()