import math
import operator
import random
import time
import sys
import threading

from enviorment.tetris import Tetris
from nat_selection.model import Model


class Agent():
    def __init__(self, cores=32, population=1000, selectPercent=0.1, mutationChance=0.05, trainingGames=5, maxTrainingMoves=100, replacePercent=0.3):
        self.population = population
        self.selectPercent = selectPercent
        self.mutationChance = mutationChance
        self.trainingGames = trainingGames
        self.maxTrainingMoves = maxTrainingMoves
        self.replacePercent = replacePercent
        self.sem = threading.Semaphore(cores)

    def __normalize(self, heightWeight, clearedWeight, holesWeight, evennessWeight):
        norm = math.sqrt(heightWeight ** 2 + clearedWeight **
                         2 + holesWeight ** 2 + evennessWeight ** 2)
        heightWeight /= norm
        clearedWeight /= norm
        holesWeight /= norm
        evennessWeight /= norm

        return Model(heightWeight, clearedWeight, holesWeight, evennessWeight)

    def __randomCandidate(self):
        return self.__normalize(random.random() - 0.5, random.random() - 0.5, random.random() - 0.5, random.random() - 0.5)

    def __calculateFitnessThread(self, candidate, games, maxMoves, render=False):
        self.sem.acquire()
        try:
            env = env = Tetris({'reduced_shapes': 0})
            totalScore = 0
            for _ in range(games):
                _, reward, done, _ = env.reset()
                score = 0

                for _ in range(maxMoves):
                    if render:
                        env.render()
                        time.sleep(0.1)

                    actions = candidate.best(env)

                    for action in actions:
                        _, reward, done, _ = env.step(action)
                        score += reward

                        if done:
                            break

                totalScore += score

            candidate.fitness = totalScore
        finally:
            self.sem.release()

    def __calculateFitness(self, candidates, games, maxMoves, render=False):
        msg = "0/" + str(len(candidates))
        sys.stdout.write(msg)
        sys.stdout.flush()

        threads = []
        for i in range(len(candidates)):
            thread = threading.Thread(target=self.__calculateFitnessThread, args=(
                candidates[i], games, maxMoves, render))
            threads.append(thread)
            thread.start()

        for i in range(len(threads)):
            threads[i].join()
            for _ in range(len(msg)):
                sys.stdout.write('\b')
            msg = str(i+1) + "/" + str(len(candidates))
            sys.stdout.write(msg)
            sys.stdout.flush()
        return sorted(candidates, key=operator.attrgetter("fitness"))

    def __mutate(self, candidate):
        rand = random.randint(0, 3)
        changeVal = random.random() * 0.4 - 0.2
        if rand == 0:
            candidate.heightWeight += changeVal
        elif rand == 1:
            candidate.clearedWeight += changeVal
        elif rand == 2:
            candidate.holesWeight += changeVal
        elif rand == 3:
            candidate.evennessWeight += changeVal

    def __tournamentSelectPair(self, candidates, selectPercent):
        randIndex = random.randint(
            0, int(len(candidates) * (1-selectPercent)) - 1)
        randPool = candidates[randIndex:int(
            randIndex+(len(candidates)*selectPercent)) + 1]
        return randPool[-1], randPool[-2]

    def __crossOver(self, candidate1, candidate2):
        if candidate1.fitness == 0 and candidate2.fitness == 0:
            candidate = Model((candidate1.heightWeight + candidate2.heightWeight)/2, (candidate1.clearedWeight + candidate2.clearedWeight)/2,
                              (candidate1.holesWeight + candidate2.holesWeight)/2, (candidate1.evennessWeight + candidate2.evennessWeight)/2)
        else:
            candidate = Model(candidate1.fitness * candidate1.heightWeight + candidate2.fitness * candidate2.heightWeight, candidate1.fitness * candidate1.clearedWeight + candidate2.fitness * candidate2.clearedWeight,
                              candidate1.fitness * candidate1.holesWeight + candidate2.fitness * candidate2.holesWeight, candidate1.fitness * candidate1.evennessWeight + candidate2.fitness * candidate2.evennessWeight)

        return self.__normalize(*candidate.getWeights())

    def train(self, epochs, render=False):
        candidates = []

        for _ in range(self.population):
            candidates.append(self.__randomCandidate())

        print("Training initial candidates...")
        candidates = self.__calculateFitness(
            candidates, self.trainingGames, self.maxTrainingMoves, render)

        for epoch in range(epochs):
            newCandidates = []

            print("\nEpoch {}".format(epoch + 1))
            print("Training new candidates...")
            # Create upto 30% of population size
            for _ in range(int(len(candidates) * self.replacePercent)):
                candidate = self.__crossOver(
                    *self.__tournamentSelectPair(candidates, self.selectPercent))

                if random.random() < self.mutationChance:
                    self.__mutate(candidate)
                candidate = self.__normalize(*candidate.getWeights())

                newCandidates.append(candidate)

            newCandidates = self.__calculateFitness(
                newCandidates, self.trainingGames, self.maxTrainingMoves)

            candidates[:len(newCandidates)] = newCandidates
            candidates = sorted(candidates, key=operator.attrgetter("fitness"))

            print("Average fitness {}".format(
                sum([i.fitness for i in candidates]) / len(candidates)))
            print("Average weights\nheight {} - cleared {} - holes {} - evenness {}".format(*
                                                                                            [sum([i.getWeights()[j] for i in candidates]) / len(candidates) for j in range(len(candidates[0].getWeights()))]))
            print("Highest fitness {}".format(candidates[-1].fitness))
            print("Highest fitness weights\nheight {} - cleared {} - holes {} - evenness {}".format(
                *candidates[-1].getWeights()))
        # Return best candidate
        return candidates[-1]
