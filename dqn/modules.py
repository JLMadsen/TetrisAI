import torch.nn as nn

class Resize(nn.Module):
    
    def __init__(self, *args): 
        super(Resize, self).__init__() 
        self.shape = args
    
    def forward(self, x): 
        return x.reshape(self.shape) 

class Print_shape(nn.Module):
    def __init__(self, message=''):
        super(Print_shape, self).__init__()
        self.message = message

    def forward(self, x):
        print(self.message+str(x.shape))
        return x