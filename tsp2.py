import sys
import time
import numpy as np
import random
from heapq import heappush, heappop
from tsp import TSPSolver
from copy import deepcopy
import networkx as nx

class LocalSearchTSPSolver(TSPSolver):
    def mst_cost(self, remaining_nodes):
        if len(remaining_nodes) <= 1:
            return 0
        G = nx.Graph()
        for i in remaining_nodes:
            for j in remaining_nodes:
                if i != j:
                    G.add_edge(i, j, weight=self.adjacency_matrix[i][j])
        mst = nx.minimum_spanning_tree(G, algorithm='prim')
        total_cost = mst.size(weight='weight')
        return total_cost

    def heuristic_mst(self, path):
        remaining_nodes = [i for i in range(self.n) if i not in path]
        curr = path[-1]
        remaining_nodes.append(curr)
        return self.mst_cost(remaining_nodes)    
    
    def hill_climbing(self, restarts=1):
        def get_neighbors(path):
            neighbors = []
            for i in range(1, len(path) - 1):
                for j in range(i + 1, len(path) - 1):
                    neighbor = path[:]
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbors.append(neighbor)
                if i == 1 and j == len(path) - 2:
                    break
            return neighbors
        best_path = None
        best_cost = float('inf')
        for _ in range(restarts):
            current_path = list(range(self.n))
            random.shuffle(current_path)
            current_path.append(current_path[0])
            current_cost = self.calculate_path_cost(current_path)
            while True:
                neighbors = get_neighbors(current_path)
                next_path = min(neighbors, key=self.calculate_path_cost)
                next_cost = self.calculate_path_cost(next_path)
                if next_cost >= current_cost:
                    break
                current_path, current_cost = next_path, next_cost

            if current_cost < best_cost:
                best_path, best_cost = current_path, current_cost

        return best_path, best_cost, None

    def simulated_annealing(self, restarts=1, initial_temp=1000, cooling_rate=0.995, absolute_temp=0.1):
        def probability_acceptance(old_cost, new_cost, temp):
            if new_cost < old_cost:
                return 1.0
            return np.exp((old_cost - new_cost) / temp)

        best_path = None
        best_cost = float('inf')
        for _ in range(restarts):
            current_path = list(range(self.n))
            random.shuffle(current_path)
            current_path.append(current_path[0])
            current_cost = self.calculate_path_cost(current_path)
            temp = initial_temp         
            while temp > absolute_temp:
                new_path = current_path[:]
                i, j = random.sample(range(1, self.n), 2)
                new_path[i], new_path[j] = new_path[j], new_path[i]
                new_cost = self.calculate_path_cost(new_path)
                if probability_acceptance(current_cost, new_cost, temp) > random.random():
                    current_path, current_cost = new_path, new_cost
                temp *= cooling_rate
            
            if current_cost < best_cost:
                best_path, best_cost = current_path, current_cost
        
        return best_path, best_cost, None


    def genetic_algorithm(self, restarts=1, population_size=100, generations=500, mutation_rate=0.01):
        def create_individual():
            individual = list(range(self.n))
            random.shuffle(individual)
            return individual

        def crossover(parent1, parent2):
            cut = random.randint(0, self.n - 1)
            child1 = parent1[:cut] + [gene for gene in parent2 if gene not in parent1[:cut]]
            child2 = parent2[:cut] + [gene for gene in parent1 if gene not in parent2[:cut]]
            return child1, child2

        def mutate(individual):
            if random.random() < mutation_rate:
                i, j = random.sample(range(self.n), 2)
                individual[i], individual[j] = individual[j], individual[i]

        def select_parents(population):
            return random.choices(population, weights=[1/self.calculate_path_cost(ind) for ind in population], k=2)

        best_path = None
        best_cost = float('inf')

        for _ in range(restarts):
            population = [create_individual() for _ in range(population_size)]
            for _ in range(generations):
                new_population = []
                for _ in range(population_size // 2):
                    parent1, parent2 = select_parents(population)
                    child1, child2 = crossover(parent1, parent2)
                    mutate(child1)
                    mutate(child2)
                    new_population.extend([child1, child2])
                population = new_population

            current_best_individual = min(population, key=self.calculate_path_cost)
            current_best_path = current_best_individual + [current_best_individual[0]]
            current_best_cost = self.calculate_path_cost(current_best_path)
            if current_best_cost < best_cost:
                best_path, best_cost = current_best_path, current_best_cost

        return best_path, best_cost, None

    def calculate_path_cost(self, path):
        temp = sum(self.adjacency_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
        temp = temp + self.adjacency_matrix[path[(len(path)) - 1]][path[0]]
        return temp

    def solve(self):
        algorithms = [self.hill_climbing, self.simulated_annealing, self.genetic_algorithm, lambda: self.a_star_search(self.heuristic_mst)]
        algorithm_names = ['hill_climbing', 'simulated_annealing', 'genetic_algorithm', 'a_star_mst']

        results = []
        for i in range(3):
            start_time = time.time()
            cpu_start_time = time.process_time()
            path, cost, _ = algorithms[i]()
            end_time = time.time()
            cpu_end_time = time.process_time()
            real_time = end_time - start_time
            cpu_time = cpu_end_time - cpu_start_time
            results.append((algorithm_names[i], path, cost, real_time, cpu_time))
            print(f"{algorithm_names[i]}: Path={path}: Cost={cost}: Real Time={real_time}: Cpu Time={cpu_time}")
        return results

if __name__ == "__main__":
    input_stream = sys.stdin
    adjacency_matrix = TSPSolver.read_adjacency_matrix(input_stream)
    solver = LocalSearchTSPSolver(adjacency_matrix)
    solver.solve()
