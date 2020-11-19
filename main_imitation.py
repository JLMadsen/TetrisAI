import csv
import numpy as np
import time

from enviorment.tetris import Tetris
from Imitation.data_handler import *
from Imitation.agent import *
from nat_selection.agent import Agent as NatAgent
from nat_selection.model import Model

env = Tetris({'reduced_shapes': 1})
model = imitation_agent(env)

learning_rate = 0.01
epochs = 30000

def train():

    x_train, y_train = read_data("train_nat3.csv")
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()

    x_test, y_test = read_data("test_nat3.csv")
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()

    batches = 600
    x_train_batches = torch.split(x_train, batches)
    y_train_batches = torch.split(y_train, batches)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(epochs):
        
        if not epoch%(epochs//100): 
            print('\nTraining: '+ str(round(epoch/epochs*100, 2)) +' %')

            for batch in range(len(x_train_batches)):
                model.loss(x_train_batches[batch], y_train_batches[batch]).backward() 
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


def generate_data(moves):
    agent = NatAgent(cores=4)

    candidate = Model([-0.8995652940240592, 0.06425443268253492, -0.3175211096545741, -0.292974392382306])

    state, reward, done, info = env.reset()

    for move in range(moves):

        action = candidate.best(env)
        for a in action:
            write_data(state, a)
            state, reward, done, info = env.step(a)


if __name__ == "__main__":
    train()
    model.save_weights()
    #model.load_weights("_60k_0.1_nat2_600")
    main()
    #generate_data(40000)
