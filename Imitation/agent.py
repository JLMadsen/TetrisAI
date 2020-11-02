import torch
import torch.nn as nn
import torchvision

class imitation_agent(nn.Module):
    def __init__(self, env):
        super(imitation_agent, self).__init__()

        self.env = env

        # Model layers (includes initialized model variables):
        self.conv = nn.Conv2d(2, 32, (20, 10))
        self.conv2 = nn.Conv2d(32, 64, (1, 5))
        self.dense = nn.Linear(64, env.action_space)
        self.ReLU = nn.ReLU()
        

    def logits(self, x):

        print(1, x.shape)
        x = self.conv(x)
        print(2, x.shape)
        x = self.ReLU(x)
        #x = self.conv2(x)
        print(3, x.shape)
        #x = self.ReLU(x)
        x = self.dense(x)
        #x = self.ReLU(x)
        print(4, x.shape)
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