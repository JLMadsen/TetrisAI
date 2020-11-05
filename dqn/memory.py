import random

class Memory:
    
    def __init__(self, size=10_000):
        self.size = size
        self.memory = []
        
    def append(self, experience):
        if len(self.memory) == self.size:
            del self.memory[0]
        self.memory.append(experience)
        
    def sample(self, num=1):
        return random.sample(self.memory, num)
    
    def clear(self):
        self.memory = []
        
    def __len__(self):
        return len(self.memory)