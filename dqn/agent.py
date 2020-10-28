class DQNAgent():
    
    def __init__(self, env):
        
        self.env = env
        
        print(env.action_space)