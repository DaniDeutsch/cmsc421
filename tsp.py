import sys
import time
import numpy as np
import random
from heapq import heappush, heappop

class TSPSolver:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.n = len(adjacency_matrix)

    @staticmethod
    def read_adjacency_matrix(input_stream):
        n = int(input_stream.readline().strip())
        matrix = []
        for _ in range(n):
            row = list(map(int, input_stream.readline().strip().split(',')))
            matrix.append(row)
        return np.array(matrix)

    def heuristic_0(self, path):
        return 0

    def heuristic_random(self, path):
        remaining_nodes = [i for i in range(self.n) if i not in path]
        curr = path[-1]
        cost = 0
        while remaining_nodes:
            next = random.choice(remaining_nodes)
            remaining_nodes.remove(next)
            cost += self.adjacency_matrix[curr][next]
            curr = next
        return cost

    def heuristic_cheapest(self, path):
        remaining_nodes = [i for i in range(self.n) if i not in path]
        curr = path[-1]
        cost = 0
        while remaining_nodes:
            next = min(remaining_nodes, key=lambda x: self.adjacency_matrix[curr][x])
            remaining_nodes.remove(next)
            cost += self.adjacency_matrix[curr][next]
            curr = next
        return cost
    
    def a_star_search(self, heuristic):
        count = 0
        start = (0,)
        frontier = []
        heappush(frontier, (0, 0, start))
        best_path = None
        while frontier:
            f, cost, path = heappop(frontier)
            if len(path) == self.n or count == 10000:
                total_cost = cost + self.adjacency_matrix[path[-1]][0]
                best_path = path + (0,)
                return best_path, total_cost, count
            for next_node in range(self.n):
                if next_node not in path:
                    new_path = path + (next_node,)
                    new_cost = cost + self.adjacency_matrix[path[-1]][next_node]
                    new_f = new_cost + heuristic(new_path)
                    heappush(frontier, (new_f, new_cost, new_path))
            count += 1

    def solve(self, heuristic):
        heuristics = [self.heuristic_0, self.heuristic_random, self.heuristic_cheapest]
        heuristic_names = ['uniform_cost', 'random_edges', 'cheapest_edges']

        results = []
        start_time = time.time()
        cpu_start_time = time.process_time()
        path, cost, expanded = self.a_star_search(heuristics[heuristic])
        cpu_end_time = time.process_time()
        end_time = time.time()
        cpu_time = cpu_end_time - cpu_start_time
        real_time = end_time - start_time
        results.append((heuristic_names[heuristic], path, cost, cpu_time, real_time))
        print(f"{heuristic_names[heuristic]}: Path={path}: Cost={cost}: CPU Time={cpu_time}: Real Time={real_time}: Nodes Expanded={expanded}")
        return results
    
    def a_uniformCost(self):
       return self.solve(0)
    def a_randomEdge(self):
        return self.solve(1)
    def a_cheapestEdge(self):
        return self.solve(2)

if __name__ == "__main__":
    input_stream = sys.stdin
    adjacency_matrix = TSPSolver.read_adjacency_matrix(input_stream)
    solver = TSPSolver(adjacency_matrix)
    solver.a_uniformCost()
    solver.a_randomEdge()
    solver.a_cheapestEdge()
