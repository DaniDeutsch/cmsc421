import subprocess
import numpy as np
import matplotlib.pyplot as plt
import csv

def generate_random_tsp_matrix(size):
    matrix = np.random.randint(1, 100, size=(size, size))
    np.fill_diagonal(matrix, 0)
    return (matrix + matrix.T) // 2

def write_matrix_to_file(matrix, filename):
    with open(filename, 'w') as f:
        f.write(f"{len(matrix)}\n")
        for row in matrix:
            f.write(','.join(map(str, row)) + '\n')

def run_tsp_on_file(filename):
    result = subprocess.run(['python3', 'tsp.py'], input=open(filename).read(), text=True, capture_output=True)
    return result.stdout

def parse_results(output):
    results = {}
    for line in output.strip().split('\n'):
        parts = line.split(': ')
        if len(parts) != 6:
            continue
        name = parts[0]
        results[name] = {
            'cost': float(parts[2].split('=')[1]),
            'nodes_expanded': int(parts[5].split('=')[1]),
            'cpu_time': float(parts[3].split('=')[1]),
            'real_time': float(parts[4].split('=')[1]),
        }
    return results

def calculate_statistics(data):
    avg = np.mean(data)
    std_dev = np.std(data)
    maximum = np.max(data)
    minimum = np.min(data)
    return avg, std_dev, maximum, minimum

def plot_combined_graphs(sizes, all_results, metric1, metric2, title, filename):
    fig, ax1 = plt.subplots()
    heuristic_names = all_results[0][0].keys()
    for heuristic_name in heuristic_names:
        values1 = [calculate_statistics([results[heuristic_name][metric1] for results in size_results])[0] for size_results in all_results]
        values2 = [calculate_statistics([results[heuristic_name][metric2] for results in size_results])[0] for size_results in all_results]
        ax1.plot(sizes, values1, label=f'{heuristic_name} {metric1.replace("_", " ").capitalize()} (Cost)')
        ax1.plot(sizes, values2, label=f'{heuristic_name} {metric2.replace("_", " ").capitalize()} (Nodes)')

    ax1.set_xlabel('Graph Size')
    ax1.set_ylabel(f'{metric1.replace("_", " ").capitalize()} and {metric2.replace("_", " ").capitalize()}')
    ax1.legend()
    plt.title(title)
    plt.savefig(f'{filename}.png')
    plt.show()

def plot_runtime_graphs(sizes, all_results, title, filename):
    fig, ax1 = plt.subplots()
    heuristic_names = all_results[0][0].keys()
    for heuristic_name in heuristic_names:
        values_cpu = [calculate_statistics([results[heuristic_name]['cpu_time'] for results in size_results])[0] for size_results in all_results]
        values_real = [calculate_statistics([results[heuristic_name]['real_time'] for results in size_results])[0] for size_results in all_results]
        ax1.plot(sizes, values_cpu, label=f'{heuristic_name} CPU Time')
        ax1.plot(sizes, values_real, label=f'{heuristic_name} Real Time')
    
    ax1.set_xlabel('Graph Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    plt.title(title)
    plt.savefig(f'{filename}.png')
    plt.show()

def write_averages_to_csv(sizes, all_results, heuristic_name, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Graph Size', 'Average Cost', 'Cost Std Dev', 'Cost Max', 'Cost Min', 
                         'Average Nodes Expanded', 'Nodes Std Dev', 'Nodes Max', 'Nodes Min', 
                         'Average CPU Time', 'CPU Std Dev', 'CPU Max', 'CPU Min', 
                         'Average Real Time', 'Real Time Std Dev', 'Real Time Max', 'Real Time Min'])
        for i, size in enumerate(sizes):
            cost_stats = calculate_statistics([results[heuristic_name]['cost'] for results in all_results[i]])
            nodes_stats = calculate_statistics([results[heuristic_name]['nodes_expanded'] for results in all_results[i]])
            cpu_stats = calculate_statistics([results[heuristic_name]['cpu_time'] for results in all_results[i]])
            real_stats = calculate_statistics([results[heuristic_name]['real_time'] for results in all_results[i]])
            writer.writerow([size, *cost_stats, *nodes_stats, *cpu_stats, *real_stats])

def main():
    sizes = [i for i in range(5, 155, 5)]
    num_graphs = 30
    all_results = []
    for size in sizes:
        size_results = []
        for _ in range(num_graphs):
            matrix = generate_random_tsp_matrix(size)
            filename = 'temp_matrix.txt'
            write_matrix_to_file(matrix, filename)
            output = run_tsp_on_file(filename)
            results = parse_results(output)
            size_results.append(results)
        all_results.append(size_results)

    heuristic_names = all_results[0][0].keys()
    for heuristic_name in heuristic_names:
        write_averages_to_csv(sizes, all_results, heuristic_name, f'{heuristic_name}_averages.csv')
    plot_combined_graphs(sizes, all_results, 'cost', 'nodes_expanded', 'Total Cost and Nodes Expanded', 'cost_nodes')
    plot_runtime_graphs(sizes, all_results, 'CPU and Real-world Runtime', 'runtime')

if __name__ == "__main__":
    main()
