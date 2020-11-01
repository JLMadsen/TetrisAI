from enviorment.tetris import Tetris

import csv
import numpy as np
import time
from Imitation.data_handler import *
from Imitation.agent import *

env = Tetris({'reduced_shapes': 1})

def train():

    model = imitation_agent(env)

    x_train, y_train = read_data("1.csv")
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()

    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    for epoch in range(3):
        model.loss(x_train, y_train).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step
 

    #print("accuracy = %s" % model.accuracy(x_test, y_test))

def main(manual=1):

    if manual:
        while 1:
            env.reset()
            done = False
            while not done:
                state, action, done = env.render(1)
                if state and action:
                    write_data(state, action)
    else:
        scores = []
        epoch = 100_000
        
        for e in range(epoch):
            
            if not e%500:
                print(e)
            
            score = 0
            state, reward, done, info = env.reset()
            
            while not done:
                
                action = env.action_sample             
                state, reward, done, info = env.step(action)
                
                env.render()
                time.sleep(0.1 if e < 2 else 0)
                
                score += reward
                
            if score != 0:
                scores.append(score)
                
        print(scores)

if __name__ == "__main__":
    train()
    main()
