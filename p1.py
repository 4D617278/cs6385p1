#!/usr/bin/env python3
import numpy as np

N = 21 # number of nodes
id = '648274120264827412026'
test_min_r = 3
test_max_r = 14 + 1

def min_net(costs, min_flow):
    for i in range(costs.shape[0]): 
        np.minimum(costs, costs[:, i, np.newaxis] + costs[i], out=costs)
        

def init(costs, min_flow, r):
    for i in range(len(id)):
        for i2 in range(len(id)):
            min_flow[i][i2] = int(id[i]) - int(id[i2])

    rng = np.random.default_rng()
    costs.fill(100)
    rand_set = tuple(range(costs.shape[0]))

    for i in range(costs.shape[0]):
        indices = rng.choice(rand_set[:i] + rand_set[i + 1:], r, replace=False)
        costs[i][indices] = 1

def show_net(net):
    print(net)

def main():
    min_flow = np.empty(shape=(N, N), dtype=int)
    costs = np.empty(shape=(N, N), dtype=int)

    for r in range(test_min_r, test_max_r):
        init(costs, min_flow, r)
        print(costs)
        print(min_flow)
        net = min_net(costs, min_flow)
        show_net(net)

if __name__ == '__main__':
    main()
