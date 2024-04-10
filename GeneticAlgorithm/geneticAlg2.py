from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def calculateFitness(pop, items, capacity):
    pop_fitness = 0
    for individual in pop:
        pop_fitness += fitness(items, capacity, individual)

    fitness_distribution = []
    for individual in pop:
        fitness_distribution.append(fitness(items, capacity, individual)/pop_fitness)


    return fitness_distribution

def tournamentSelection(pop, n, items, capacity):
    parents=[]
    for _ in range(n):
        parent1 = random.choice(pop)
        parent2 = random.choice(pop)
        if fitness(items, capacity, parent1) >= fitness(items, capacity, parent2):
            parents.append(parent1)
        else:
            parents.append(parent2)
    return parents
    

def createChildren(parent1, parent2):
    points = []
    points.append(random.randint(0, len(parent1) - 1))
    points.append(random.randint(0, len(parent1) - 1))
    points.sort()
    child1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
    child2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]
    return child1, child2

def crossover(pop, size, n_elite):
    new_generation = []
    for i in range((size-n_elite)//2):
        children = list(createChildren(random.choice(pop), random.choice(pop)))
        new_generation += children

    if size-n_elite % 2 != 0:
        child1, child2 = createChildren(random.choice(pop), random.choice(pop))
        new_generation.append(child1)

    return new_generation

def mutate(pop, mutation_rate):
    for individual in pop:
        for i in range(len(individual)):
            if random.random() <= mutation_rate:
                individual[i] = not individual[i]

    return pop

def chooseElites(pop, fitness, n):
    sorted_pop = sorted(pop, key= lambda x: fitness[pop.index(x)])
    sorted_pop.reverse()
    return sorted_pop[:n]



items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_elite = 1
n_selection = population_size - n_elite

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm

    # step 1 - choose parents
    parents = tournamentSelection(population, n_selection, items, knapsack_max_capacity)

    # step 2 - create a new generation
    new_generation = crossover(parents, population_size, n_elite)

    # step 3 - mutate
    new_generation = mutate(new_generation, 0.01)

    # step 4 - update the population
    fitness_distribution = calculateFitness(population, items, knapsack_max_capacity)
    elites = chooseElites(population, fitness_distribution, n_elite)
    new_generation.extend(elites)
    population = new_generation


    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()