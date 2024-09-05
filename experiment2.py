import numpy as np
import csv
import subprocess
from experiment import write_matrix_to_file
import matplotlib.pyplot as plt

def generate_tsp_matrix_with_known_optimal(n, optimal_cost):
    cycle_cost = optimal_cost // (n)
    matrix = np.random.randint(cycle_cost, cycle_cost + min(n, 10), size=(n, n))
    matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(matrix, 0)
    cycle = list(range(n)) + [0]
    for i in range(n):
        matrix[cycle[i], cycle[i + 1]] = cycle_cost
        matrix[cycle[i + 1], cycle[i]] = cycle_cost   
    return matrix

def run_tsp_on_file(filename):
    result = subprocess.run(['python3', 'tsp2.py'], input=open(filename).read(), text=True, capture_output=True)
    return result.stdout

def parse_results(output):
    lines = output.strip().split('\n')
    results = {}
    for line in lines:
        parts = line.split(': ')
        algorithm = parts[0]
        cost = float(parts[2].split('=')[1])
        real_time = float(parts[3].split('=')[1])
        cpu_time = float(parts[4].split('=')[1])
        results[algorithm] = {
            'cost': cost,
            'cpu_time': cpu_time,
            'real_time': real_time
        }
    return results

def write_experiment_results(filename, all_results, optimal_costs):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Graph Size', 'Algorithm', 'Total Cost', 'Optimal Cost', 'Cost Difference', 'CPU Time', 'Real Time'])
        for size, results_list in all_results.items():
            for results in results_list:
                for algorithm, metrics in results.items():
                    optimal_cost = optimal_costs[size]
                    cost_diff = metrics['cost'] - optimal_cost
                    writer.writerow([size, algorithm, metrics['cost'], optimal_cost, cost_diff, metrics['cpu_time'], metrics['real_time']])

def plot_graphs(sizes, all_results, optimal_costs):
    algorithms = ['hill_climbing', 'simulated_annealing', 'genetic_algorithm']
    
    for algo in algorithms:
        cpu_times_avg = []
        cost_diffs_avg = []
        cost_diffs_min = []
        cost_diffs_max = []        
        for size in sizes:
            algo_results = [results[algo] for results in all_results[size]]
            optimal_cost = optimal_costs[size]
            cpu_times_per_size = [result['cpu_time'] for result in algo_results]
            cost_diffs_per_size = [result['cost'] - optimal_cost for result in algo_results]        
            cpu_times_avg.append(np.mean(cpu_times_per_size))
            cost_diffs_avg.append(np.mean(cost_diffs_per_size))
            cost_diffs_min.append(np.min(cost_diffs_per_size))
            cost_diffs_max.append(np.max(cost_diffs_per_size))

        plt.figure()
        plt.plot(cpu_times_avg, cost_diffs_avg, label=f'{algo} Average', color='blue', marker='o')
        plt.scatter(cpu_times_avg, cost_diffs_min, label=f'{algo} Min', color='green', marker='x')
        plt.scatter(cpu_times_avg, cost_diffs_max, label=f'{algo} Max', color='red', marker='s')
        plt.xlabel('Average CPU Time')
        plt.ylabel('Cost Difference')
        plt.title(f'Cost Difference vs CPU Time for {algo}')
        plt.legend()
        plt.savefig(f'{algo}_cost_diff_vs_cpu_time.png')
        plt.show()

def run_experiment():
    sizes = [i for i in range(5, 155, 5)]
    num_graphs = 5
    all_results = {size: [] for size in sizes}
    optimal_costs = {}

    for size in sizes:
        optimal_cost = size * (size)
        optimal_costs[size] = optimal_cost

        for i in range(num_graphs):
            matrix = generate_tsp_matrix_with_known_optimal(size, optimal_cost)
            filename = f'temp_matrix_2.txt'
            write_matrix_to_file(matrix, filename)
            output = run_tsp_on_file(filename)
            results = parse_results(output)
            all_results[size].append(results)

    write_experiment_results('experiment2_results.csv', all_results, optimal_costs)
    plot_graphs(sizes, all_results, optimal_costs)

def main():
    run_experiment()

if __name__ == "__main__":
    main()
