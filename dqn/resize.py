import torch.nn as nn

class Resize(nn.Module):
    
    def __init__(self, *args): 
        super(Resize, self).__init__() 
        self.shape = args
    
    def forward(self, x): 
        return x.reshape(self.shape) 
