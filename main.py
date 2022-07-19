# -*- coding: utf-8 -*-

import abc
import dataclasses

from numpy.random import randint, rand
from progress.bar import PixelBar as Bar
from random import choice
from typing import Callable


def draw_probability_rate(rate):
    return rand() < rate


def flip_the_bit(bin_value):
    return 1 - bin_value


def pairwise_iterator(seq):
    a = iter(seq)
    return zip(a, a)


def compare_minimize(a, b) -> bool:
    return a < b

def compare_maximize(a, b) -> bool:
    return a > b



class AbstractChromosome(abc.ABC):
    @property
    @abc.abstractmethod
    def content(self):
        pass

    @abc.abstractmethod
    def mutate(self):
        pass

    @classmethod
    @abc.abstractmethod
    def crossover(cls, parent_1, parent_2, crossover_rate):
        pass


@dataclasses.dataclass
class BinaryChromosome(AbstractChromosome):
    vector: list[int] = dataclasses.field(compare=False)
    mutation_rate: float = dataclasses.field(compare=False)

    @property
    def content(self):
        return self.vector

    def __len__(self) -> int:
        return len(self.vector)

    def __setitem__(self, index, data):
        self.vector[index] = data

    def __getitem__(self, index):
        return self.vector[index]

    def mutate(self):
        for index, elem in enumerate(self.vector):
            if draw_probability_rate(self.mutation_rate):
                self.vector[index] = flip_the_bit(elem)

    @classmethod
    def crossover(cls, parent_1, parent_2, crossover_rate):
        child_1 = dataclasses.replace(parent_1)
        child_2 = dataclasses.replace(parent_2)
        if draw_probability_rate(crossover_rate):
            pivot = randint(
                1, len(child_1) - 2
            )  # get crossover point that is neither begining nor the end
            child_1.vector = parent_1.vector[:pivot] + parent_2.vector[pivot:]
            child_2.vector = parent_2.vector[:pivot] + parent_1.vector[pivot:]
        return child_1, child_2


@dataclasses.dataclass
class ContinuousChromosome(AbstractChromosome):
    vector: list[int] = dataclasses.field(compare=False)
    mutation_rate: float = dataclasses.field(compare=False)
    bounds: list = dataclasses.field(compare=False)

    @property
    def content(self):
        decoded = []
        n_bits = len(self.vector) // len(self.bounds)
        largest = 2**n_bits
        for i in range(len(self.bounds)):
            # extract the substring
            start, end = i * n_bits, (i * n_bits)+n_bits
            substring = self.vector[start:end]
            # convert bitstring to a string of chars
            chars = ''.join([str(s) for s in substring])
            # convert string to integer
            integer = int(chars, 2)
            # scale integer to desired range
            value = self.bounds[i][0] + (integer/largest) * (self.bounds[i][1] - self.bounds[i][0])
            # store
            decoded.append(value)
        return decoded


    def __len__(self) -> int:
        return len(self.vector)

    def __setitem__(self, index, data):
        self.vector[index] = data

    def __getitem__(self, index):
        return self.vector[index]

    def mutate(self):
        for index, elem in enumerate(self.vector):
            if draw_probability_rate(self.mutation_rate):
                self.vector[index] = flip_the_bit(elem)

    @classmethod
    def crossover(cls, parent_1, parent_2, crossover_rate):
        child_1 = dataclasses.replace(parent_1)
        child_2 = dataclasses.replace(parent_2)
        if draw_probability_rate(crossover_rate):
            pivot = randint(
                1, len(child_1) - 2
            )  # get crossover point that is neither begining nor the end
            child_1.vector = parent_1.vector[:pivot] + parent_2.vector[pivot:]
            child_2.vector = parent_2.vector[:pivot] + parent_1.vector[pivot:]
        return child_1, child_2


class AbstractChromosomeRanker(abc.ABC):
    @abc.abstractmethod
    def score(self, chromosome: AbstractChromosome) -> float:
        pass


class BinaryVectorRanker(AbstractChromosomeRanker):
    def score(self, chromosome: BinaryChromosome) -> float:
        return sum(chromosome.content)


class ContinuousFunctionRanker(AbstractChromosomeRanker):
    def score(self, chromosome: ContinuousChromosome):
        return chromosome.content[0]**2.0 + chromosome.content[1]**2.0


def get_random_binary_chromosome(
    chromosome_length: int,
    mutation_rate: float,
    bounds: list | None = None
) -> list[BinaryChromosome]:
    return BinaryChromosome(randint(0, 2, chromosome_length).tolist(), mutation_rate)


def get_random_continuous_chromosome(
    chromosome_length: int,
    mutation_rate: float,
    bounds: list | None = None
) -> list[BinaryChromosome]:
    return ContinuousChromosome(randint(0, 2, chromosome_length*len(bounds)).tolist(), mutation_rate, bounds)


@dataclasses.dataclass(slots=True)
class GeneticAlgorithm:
    chromosome_ranker: AbstractChromosomeRanker
    chromosome_length: int
    chromosome_factory: Callable
    number_of_generations: int
    population_size: int
    crossover_rate: float
    crossover_factory: Callable
    mutation_rate: float
    comparator: Callable
    bounds: list = dataclasses.field(default_factory=list)
    selection_param: int = 2

    def select_from_population(self, population: list):
        # first random selection
        selection = choice(population)
        for _ in range(self.selection_param):
            # check if better (e.g. perform a tournament)
            candidate = choice(population)
            if self.comparator(self.chromosome_ranker.score(candidate), self.chromosome_ranker.score(selection)):
                selection = candidate
        return selection

    def run(self):
        # initial population of random bitstring
        population = [
            self.chromosome_factory(self.chromosome_length, self.mutation_rate, self.bounds)
            for _ in range(self.population_size)
        ]

        # keep track of best solution
        best_candidate = population[0]

        with Bar('Evolving...', max=self.number_of_generations) as bar:
            for _ in range(self.number_of_generations):
                # check for new best solution
                # print(gen, best_candidate.content)
                for candidate in population:
                    if self.comparator(self.chromosome_ranker.score(candidate), self.chromosome_ranker.score(best_candidate)):
                        best_candidate = candidate

                # select parents
                selected_parents = [
                    self.select_from_population(population) for _ in range(self.population_size)
                ]

                # create the next population
                next_population = []

                for p1, p2 in pairwise_iterator(selected_parents):
                    # crossover and mutation
                    for offspring in self.crossover_factory(p1, p2, self.crossover_rate):
                        offspring.mutate()
                        next_population.append(offspring)

                # replace population
                population = next_population
                bar.next()

        return best_candidate


if __name__ == "__main__":

    # https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

    # bounds for continuous function
    bounds  = [[-5.0, 5.0], [-5.0, 5.0]]

    # define the total iterations
    number_of_generations = 200

    # number of bits in a single chromosome
    chromosome_length = 16

    # define the population_size
    population_size = 100

    # crossover_rate
    crossover_rate = 0.9

    # mutation_rate
    # mutation_rate = 1.0 / float(chromosome_length)
    mutation_rate = 1.0 / (float(chromosome_length) * len(bounds))

    ranker = ContinuousFunctionRanker()
    # ranker = BinaryVectorRanker()

    # perform the genetic algorithm search
    ga = GeneticAlgorithm(
        ranker,
        chromosome_length,
        get_random_continuous_chromosome,
        number_of_generations,
        population_size,
        crossover_rate,
        ContinuousChromosome.crossover,
        mutation_rate,
        compare_minimize,
        bounds,
    )

    print('Running: ', ga)
    print()
    best = ga.run()
    print('Done!')
    print()
    print(ranker.score(best), best.content)
