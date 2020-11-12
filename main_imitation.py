from enviorment.tetris import Tetris

import csv
import numpy as np
import time
from Imitation.data_handler import *
from Imitation.agent import *

env = Tetris({'reduced_shapes': 1})
model = imitation_agent(env)

learning_rate = 0.01
epochs = 100000

def train():

    x_train, y_train = read_data("train_jakob.csv")
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()

    x_test, y_test = read_data("test_jakob.csv")
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(epochs):
        
        if not epoch%(epochs//100): 
            print('\nTraining: '+ str(round(epoch/epochs*100, 2)) +' %')

        model.loss(x_train, y_train).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step
 

    print("accuracy = %s" % model.accuracy(x_test, y_test))

def main(manual=0):

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
                state = torch.tensor(state).unsqueeze(0).float()
                action = (model.f(state)).argmax(1)          
                state, reward, done, info = env.step(action)

                env.render()
                #time.sleep(0.1 if e < 2 else 0)
                
                if reward:
                    print(reward, e)

                score += reward
                
            if score != 0:
                scores.append(score)
                
        print(scores)

if __name__ == "__main__":
    train()
    model.save_weights()
    #model.load_weights("_hot")
    main()
