#!/usr/bin/env python3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

N = 21 # number of nodes
id = '648274120264827412026'
test_min_r = 3
test_max_r = 14 + 1

# module 2
def min_net(costs, demands, net):
    min_paths = np.empty((N, N), dtype=object)

    for i in range(N):
        for i2 in range(N):
            min_paths[i][i2] = []

    for i in range(N): 
        for i2 in range(N):
            for i3 in range(N):
                min_cost_i = costs[i2][i] + costs[i][i3]
                if costs[i2][i3] > min_cost_i:
                    min_paths[i2][i3] = min_paths[i2][i] + [i] + min_paths[i][i3]
                    costs[i2][i3] = min_cost_i

    for i in range(N):
        for i2 in range(N):
            min_path = min_paths[i][i2]
            min_path.append(i2)
            for i3 in range(len(min_path) - 1): 
                edge = (min_path[i3], min_path[i3 + 1])
                net[edge] += demands[i][i2]

# module 3
def show(densities, net, total_costs):
    plt.plot(total_costs)
    plt.xlabel('k')
    plt.ylabel('Total Cost')
    plt.savefig('costs.png')
    plt.clf()

    plt.plot(densities)
    plt.xlabel('k')
    plt.ylabel('Density')
    plt.savefig('densities.png')
    plt.clf()

    plt.figure(figsize=(20, 20))

    for i in [3, 8, 14]:
        G = nx.from_numpy_matrix(net[i])
        #layout = nx.circular_layout(G)
        layout = nx.shell_layout(G)
        nx.draw(G, layout, with_labels=True)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=edge_labels)
        plt.savefig(f'graph{i}.png')
        plt.clf()

def main():
    demands = np.empty((N, N), dtype=int)
    costs = np.full((N, N), 100)
    net = np.zeros((N, N, N), dtype=int)
    total_costs = np.zeros(N, dtype=int)
    densities = np.zeros(N)

    # module 1
    for i in range(len(id)):
        for i2 in range(len(id)):
            demands[i][i2] = max(int(id[i]) - int(id[i2]), 0)

    rng = np.random.default_rng()
    rand_set = tuple(range(N))
    num_dir_edges = N * (N - 1)

    for r in range(test_min_r, test_max_r):
        costs.fill(100)
        np.fill_diagonal(costs, 0)

        for i in range(N):
            indices = rng.choice(rand_set[:i] + rand_set[i + 1:], r, replace=False)
            costs[i][indices] = 1

        #print(costs)
        #print(demands)

        min_net(costs, demands, net[r])
        total_costs[r] = np.sum(demands * costs)
        densities[r] =  float(np.count_nonzero(net[r])) / num_dir_edges

    show(densities, net, total_costs)

if __name__ == '__main__':
    main()
