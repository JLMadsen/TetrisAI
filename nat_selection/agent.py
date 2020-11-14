import math
import operator
import random
import time
import sys
import threading
import matplotlib.pyplot as plt

from enviorment.tetris import Tetris
from nat_selection.model import Model, weightsNum


class Agent():
    def __init__(self, cores=32, population=500, selectChance=0.1, mutateChance=0.05, games=5, moves=50, replacePercent=0.3):
        self.population = population
        self.selectChance = selectChance
        self.mutateChance = mutateChance
        self.games = games
        self.moves = moves
        self.replacePercent = replacePercent
        self.sem = threading.Semaphore(cores)

    def __normalize(self, weights):
        norm = math.sqrt(sum([pow(w, 2) for w in weights]))
        return Model([*map(lambda x: x/norm, weights)])

    def __randomCandidate(self):
        return self.__normalize([(random.random() - 0.5) for _ in range(weightsNum)])

    def __calculateFitnessThread(self, candidate, games, maxMoves):
        self.sem.acquire()
        try:
            env = env = Tetris({'reduced_shapes': 0})
            totalScore = 0
            for _ in range(games):
                _, reward, done, _ = env.reset()
                score = 0

                for _ in range(maxMoves):
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

    def __calculateFitness(self, candidates, games, maxMoves):
        msg = "0/" + str(len(candidates))
        sys.stdout.write(msg)
        sys.stdout.flush()

        threads = []
        for i in range(len(candidates)):
            thread = threading.Thread(target=self.__calculateFitnessThread, args=(
                candidates[i], games, maxMoves))
            threads.append(thread)
            thread.start()

        for i in range(len(threads)):
            threads[i].join()
            for _ in range(len(msg)):
                sys.stdout.write('\b')
            msg = str(i+1) + "/" + str(len(candidates))
            sys.stdout.write(msg)
            sys.stdout.flush()
        print()
        return sorted(candidates, key=operator.attrgetter("fitness"))

    def __mutate(self, candidate):
        rand = random.randint(0, 3)
        changeVal = random.random() * 0.4 - 0.2
        candidate.weights[rand] += changeVal

    def __tournamentSelectPair(self, candidates, selectChance):
        randIndex = random.randint(
            0, int(len(candidates) * (1-selectChance)) - 1)
        randPool = candidates[randIndex:int(
            randIndex+(len(candidates)*selectChance)) + 1]
        return randPool[-1], randPool[-2]

    def __crossOver(self, candidate1, candidate2):
        if candidate1.fitness == 0 and candidate2.fitness == 0:
            candidate = Model(
                [((candidate1.weights[i] + candidate2.weights[i])/2) for i in range(weightsNum)])
        else:
            candidate = Model([(candidate1.fitness * candidate1.weights[i] + candidate2.fitness *
                                candidate2.weights[i]) for i in range(weightsNum)])
        return self.__normalize(candidate.weights)

    def train(self, generations):
        candidates = []
        avgWeights = []
        avgScores = []
        for _ in range(self.population):
            candidates.append(self.__randomCandidate())

        print("Training initial candidates...")
        candidates = self.__calculateFitness(
            candidates, self.games, self.moves)

        for gen in range(generations):
            newCandidates = []

            print("\nGeneration {}".format(gen + 1))
            print("Training new candidates...")
            # Create upto 30% of population size
            for _ in range(int(len(candidates) * self.replacePercent)):
                candidate = self.__crossOver(
                    *self.__tournamentSelectPair(candidates, self.selectChance))

                if random.random() < self.mutateChance:
                    self.__mutate(candidate)
                candidate = self.__normalize(candidate.weights)

                newCandidates.append(candidate)

            newCandidates = self.__calculateFitness(
                newCandidates, self.games, self.moves)

            candidates[:len(newCandidates)] = newCandidates
            candidates = sorted(candidates, key=operator.attrgetter("fitness"))

            avgScores.append(
                sum([i.fitness for i in candidates]) / len(candidates))
            avgWeights.append(
                [(sum([c.weights[j] for c in candidates]) / len(candidates)) for j in range(weightsNum)])

            labels = ["Sum height", "Cleared lines",
                      "Holes", "Evenness", "Max height"]
            print("\nAverage fitness {}".format(avgScores[-1]))
            print("Average weights:")
            print(" - ".join("{} {}".format(*i)
                             for i in [*zip(labels, avgWeights[-1])]))
            print("\nHighest fitness {}".format(candidates[-1].fitness))
            print("Best weights:")
            print(" - ".join("{} {}".format(*i)
                             for i in [*zip(labels, candidates[-1].weights)]))
            print(
                "--------------------------------------------------------------------------------------------")

        plt.plot([*range(len(avgScores))], avgScores)
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.show()

        plt.figure(figsize=(7, 7))
        for i in range(weightsNum):
            plt.plot([*zip(*avgWeights)][i], label=labels[i])
        plt.ylabel("Weight")
        plt.xlabel("Generation")
        plt.legend()
        plt.show()

        return candidates[-1]
