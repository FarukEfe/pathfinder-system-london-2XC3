import heapq
import copy
import matplotlib.pyplot as plt
import numpy as np
import timeit
from itertools import product
from random import sample, randint
from sys import getsizeof
import os

from Algorithms import Dijkstra, BellmanFord, SPAlgorithm
from Graphs import WeightedGraph, Graph

# MARK: Plots

def color_plot_comparison(run_arr: list[list[float]], x: list[float], labels: list[str], name: str, means: list[list[float]], colors, xlabel):
    num_runs = len(run_arr)
    plt.figure(figsize=(12, 8))
    for i in range(num_runs):
        plt.plot(x, run_arr[i], label = labels[i])
        plt.axhline(means[i], linestyle="--",label="Avg of "+labels[i], color=colors[i]) 
    plt.title('time cost comparison ' + name)  
    plt.xlabel(xlabel)  
    plt.ylabel("run time")  
    plt.legend()  
    plt.grid(True)  
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'part_2_3', f'{name}.jpg'))

def color_plot_comparison_mem(run_arr: list[list[float]], x: list[float], labels: list[str], name: str, means: list[list[float]], colors, xlabel):
    num_runs = len(run_arr)
    plt.figure(figsize=(12, 8))
    for i in range(num_runs):
        plt.plot(x, run_arr[i], label = labels[i])
        plt.axhline(means[i], linestyle="--",label="Avg of "+labels[i], color=colors[i]) 
    plt.title('space cost comparison ' + name)  
    plt.xlabel(xlabel)  
    plt.ylabel("bytes")  
    plt.legend()  
    plt.grid(True)  
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'part_2_3', f'space_{name}.jpg'))

def plot_comparison(run_arr: list[list[float]], x: list[float], name: str, labels: list[str], xname, yname, file_name):
    num_runs = len(run_arr)
    plt.figure(figsize=(20, 8))
    for i in range(num_runs):
        plt.plot(x, run_arr[i], label = labels[i])  
    plt.title('time complexity comparison ' + name)  
    plt.xlabel(xname)  
    plt.ylabel(yname)  
    plt.legend()  
    plt.grid(True)  
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(), 'part_2_3', f'{name}.jpg'))

# MARK: Garbage

def dijkstra(graph: WeightedGraph, source: int, k: int):

    dist_table = { v: float('inf') for v in graph.graph.keys() }
    prev_table = { v: -1 for v in graph.graph.keys() }
    dist_table[source], prev_table[source] = 0, source

    for _ in range(k):
        pq = [(0, source)]
        heapq.heapify(pq)
        while len(pq) > 0:
            _, u = heapq.heappop(pq)
            #print(_, u, graph.graph[u])
            # Relax
            for v in graph.graph[u]:
                #print(f'edge: {v}')
                alt = dist_table[u] + graph.w(u,v)
                if alt < dist_table[v]:
                    dist_table[v] = alt
                    prev_table[v] = u
                    heapq.heappush(pq, (alt, v))
    
    return dist_table, prev_table
    
def dijkstra_mem(graph: WeightedGraph, source: int, k: int):

    dist_table = { v: float('inf') for v in graph.graph.keys() }
    prev_table = { v: -1 for v in graph.graph.keys() }
    dist_table[source], prev_table[source] = 0, source

    mem_heap = 0
    for _ in range(k):
        pq = [(0, source)]
        heapq.heapify(pq)

        temp = getsizeof(pq)
        if temp > mem_heap:
            mem_heap = temp

        while len(pq) > 0:
            _, u = heapq.heappop(pq)
            #print(_, u, graph.graph[u])
            # Relax
            for v in graph.graph[u]:
                #print(f'edge: {v}')
                alt = dist_table[u] + graph.w(u,v)
                if alt < dist_table[v]:
                    dist_table[v] = alt
                    prev_table[v] = u
                    heapq.heappush(pq, (alt, v))
    memo = getsizeof(dist_table) + getsizeof(prev_table) + mem_heap
    return memo

def bellmanFord(graph: WeightedGraph, source: int, k: int):
    # The Algorithm Comes Here
    # Initialize the distance to all vertices as infinity
    distances = {vertex: float('inf') for vertex,_ in graph.graph.items()}
    prev = {vertex: -1 for vertex,_ in graph.graph.items()}
    distances[source] = 0

    # Relax all edges up to k times
    for _ in range(k):
        # Copy of distances to avoid interference during updates
        new_distances = copy.deepcopy(distances)

        # For each edge in the graph
        for u in graph.graph.keys():
            for v in graph.graph[u]:
                weight = graph.weights[(u, v)]
                if distances[u] + weight < new_distances[v]:
                    prev[v] = u
                    new_distances[v] = distances[u] + weight

        distances = new_distances

    return distances, prev
    
def bellmanFord_mem(graph: WeightedGraph, source: int, k: int):
    # The Algorithm Comes Here
    # Initialize the distance to all vertices as infinity
    distances = {vertex: float('inf') for vertex,_ in graph.graph.items()}
    prev = {vertex: -1 for vertex,_ in graph.graph.items()}
    distances[source] = 0
    # Relax all edges up to k times
    for _ in range(k):
        # Copy of distances to avoid interference during updates
        # For each edge in the graph
        for u in graph.graph.keys():
            for v in graph.graph[u]:
                weight = graph.weights[(u, v)]
                if distances[u] + weight < distances[v]:
                    prev[v] = u
                    distances[v] = distances[u] + weight

    memo = getsizeof(distances) + getsizeof(prev)
    return memo
    
def create_random_graph(nodes: int, edges: int, w_min: float, w_max: float) -> WeightedGraph:
        node_list = [n for n in range(nodes)]
        edge_list = list(product(node_list, repeat=2))
        edge_list = sample(edge_list, k=edges)
        graph = WeightedGraph(nodes)
        for u,v in edge_list:
            w = randint(w_min, w_max)
            graph.add_edge(u,v,w)
        return graph

def generate_best_case_weighted_graph(nodes: int) -> WeightedGraph:
    # Initialize a WeightedGraph with the given number of nodes
    best_case_graph = WeightedGraph(nodes)
    
    # Add edges to form a simple chain (0-1-2-3-4-...)
    for i in range(nodes - 1):
        best_case_graph.add_edge(i, i + 1, 1)  # Add an edge with weight 1
    
    return best_case_graph

# MARK: Experiments

# full_density_proper_k
def experiment1():
    N = 25
    nodes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    edges = [3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136]
    bf, d = [], []
    wmin, wmax = 1, 5
    for i in range(len(nodes)):
        trials = N
        temp_b = []
        temp_d = []
        while trials > 0:
            g = create_random_graph(nodes[i], edges[i], wmin, wmax)
            source = randint(0, nodes[i] - 1)
            mid = (nodes[i] - 1)//2
            k = randint(int(mid * 0.75), int(mid * 1.25))

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = dijkstra(target, source, k)
            stop = timeit.default_timer()
            temp_d.append(stop-start)

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = bellmanFord(target, source, k)
            stop = timeit.default_timer()
            temp_b.append(stop-start)
            trials -= 1

        d.append(np.mean(temp_d))
        bf.append(np.mean(temp_b))

    run_arr = [d, bf]
    xbar = nodes
    labels = ["Dijkstra", "BellmanFord"]
    means = [np.mean(d), np.mean(bf)]
    colors = ["blue", "orange"]
    # print(d, bf)
    color_plot_comparison(run_arr, xbar, labels, "full_density_proper_k", means, colors, "nodes")
    return 0

# 50_density_proper_k
def experiment2():
    N = 25
    nodes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    edges = [2, 3, 5, 8, 11, 14, 18, 23, 28, 33, 39, 46, 53, 60, 68]
    bf, d = [], []
    wmin, wmax = 1, 5
    for i in range(len(nodes)):
        trials = N
        temp_b = []
        temp_d = []
        while trials > 0:
            g = create_random_graph(nodes[i], edges[i], wmin, wmax)
            source = randint(0, nodes[i] - 1)
            mid = (nodes[i] - 1)//2
            k = randint(int(mid * 0.75), int(mid * 1.25))

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = dijkstra(target, source, k)
            stop = timeit.default_timer()
            temp_d.append(stop-start)

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = bellmanFord(target, source, k)
            stop = timeit.default_timer()
            temp_b.append(stop-start)
            trials -= 1

        d.append(np.mean(temp_d))
        bf.append(np.mean(temp_b))

    run_arr = [d, bf]
    xbar = nodes
    labels = ["Dijkstra", "BellmanFord"]
    means = [np.mean(d), np.mean(bf)]
    colors = ["blue", "orange"]
    # print(d, bf)
    color_plot_comparison(run_arr, xbar, labels, "50_density_proper_k", means, colors, "nodes")
    return 0

# nodes_test_fix_edges=20_proper_k
def experiment3():
    N = 25
    nodes = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    edges = 20
    bf, d = [], []
    wmin, wmax = 1, 5
    for i in range(len(nodes)):
        trials = N
        temp_b = []
        temp_d = []
        while trials > 0:
            g = create_random_graph(nodes[i], edges, wmin, wmax)
            source = randint(0, nodes[i] - 1)
            mid = (nodes[i] - 1)//2
            k = randint(int(mid * 0.75), int(mid * 1.25))

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = dijkstra(target, source, k)
            stop = timeit.default_timer()
            temp_d.append(stop-start)

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = bellmanFord(target, source, k)
            stop = timeit.default_timer()
            temp_b.append(stop-start)
            trials -= 1

        d.append(np.mean(temp_d))
        bf.append(np.mean(temp_b))

    run_arr = [d, bf]
    xbar = nodes
    labels = ["Dijkstra", "BellmanFord"]
    means = [np.mean(d), np.mean(bf)]
    colors = ["blue", "orange"]
    # print(d, bf)
    color_plot_comparison(run_arr, xbar, labels, "nodes_test_fix_edges=20_proper_k", means, colors, "nodes")
    return 0
# edges_test_fix_nodes=10_proper_k
def experiment4():
    N = 25
    nodes = 10
    edges = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    bf, d = [], []
    wmin, wmax = 1, 5
    for i in range(len(edges)):
        temp_b = []
        temp_d = []
        trials = N
        while trials > 0:
            g = create_random_graph(nodes, edges[i], wmin, wmax)
            source = randint(0, nodes - 1)
            mid = (nodes - 1)//2
            k = randint(int(mid * 0.75), int(mid * 1.25))

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = dijkstra(target, source, k)
            stop = timeit.default_timer()
            temp_d.append(stop-start)

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = bellmanFord(target, source, k)
            stop = timeit.default_timer()
            temp_b.append(stop-start)
            trials -= 1

        d.append(np.mean(temp_d))
        bf.append(np.mean(temp_b))

    run_arr = [d, bf]
    xbar = edges
    labels = ["Dijkstra", "BellmanFord"]
    means = [np.mean(d), np.mean(bf)]
    colors = ["blue", "orange"]
    # print(d, bf)
    color_plot_comparison(run_arr, xbar, labels, "edges_test_fix_nodes=10_proper_k", means, colors, "edges")
    return 0
# k_test_fix_nodes=10_edges=15
def experiment5():
    N = 25
    nodes = 10
    edges = 15
    bf, d = [], []
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    wmin, wmax = 1, 5
    for i in range(len(k)):
        trials = N
        temp_b = []
        temp_d = []
        while trials > 0:
            g = create_random_graph(nodes, edges, wmin, wmax)
            source = randint(0, nodes - 1)

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = dijkstra(target, source, k[i])
            stop = timeit.default_timer()
            temp_d.append(stop-start)

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = bellmanFord(target, source, k[i])
            stop = timeit.default_timer()
            temp_b.append(stop-start)
            trials -= 1

        d.append(np.mean(temp_d))
        bf.append(np.mean(temp_b))

    run_arr = [d, bf]
    xbar = k
    labels = ["Dijkstra", "BellmanFord"]
    means = [np.mean(d), np.mean(bf)]
    colors = ["blue", "orange"]
    # print(d, bf)
    color_plot_comparison(run_arr, xbar, labels, "k_test_fix_nodes=10_edges=15", means, colors, "index k")
    return 0
# k_test_best_case_of_Dijkstra
def experiment6():
    N = 10
    nodes = 10
    bf, d = [], []
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(len(k)):
        trials = N
        temp_b = []
        temp_d = []
        while trials > 0:
            g = generate_best_case_weighted_graph(nodes)
            source = randint(0, nodes - 1)

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = dijkstra(target, source, k[i])
            stop = timeit.default_timer()
            temp_d.append(stop-start)

            target = copy.deepcopy(g)
            start = timeit.default_timer()
            temp = bellmanFord(target, source, k[i])
            stop = timeit.default_timer()
            temp_b.append(stop-start)
            trials -= 1

        d.append(np.mean(temp_d))
        bf.append(np.mean(temp_b))

    run_arr = [d, bf]
    xbar = k
    labels = ["Dijkstra", "BellmanFord"]
    means = [np.mean(d), np.mean(bf)]
    colors = ["blue", "orange"]
    # print(d, bf)
    color_plot_comparison(run_arr, xbar, labels, "k_test_best_case_of_Dijkstra", means, colors, "index k")
    return 0
# space cost k_test_fix_nodes=10_edges=15
def experiment7():
    N = 25
    nodes = 10
    edges = 15
    bf, d = [], []
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    wmin, wmax = 1, 5
    for i in range(len(k)):
        trials = N
        temp_b = []
        temp_d = []
        while trials > 0:
            g = create_random_graph(nodes, edges, wmin, wmax)
            source = randint(0, nodes - 1)

            target = copy.deepcopy(g)
            temp = dijkstra_mem(target, source, k[i])
            temp_d.append(temp)

            target = copy.deepcopy(g)
            temp = bellmanFord_mem(target, source, k[i])
            temp_b.append(temp)
            trials -= 1
        d.append(np.mean(temp_d))
        bf.append(np.mean(temp_b))

    run_arr = [d, bf]
    xbar = k
    labels = ["Dijkstra", "BellmanFord"]
    means = [np.mean(d), np.mean(bf)]
    colors = ["blue", "orange"]
    # print(d, bf)
    color_plot_comparison_mem(run_arr, xbar, labels, "k_test_fix_nodes=10_edges=15", means, colors, "index k")
    return 0
# space cost 50_density_proper_k
def experiment8():
    N = 25
    nodes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    edges = [2, 3, 5, 8, 11, 14, 18, 23, 28, 33, 39, 46, 53, 60, 68]
    bf, d = [], []
    wmin, wmax = 1, 5
    for i in range(len(nodes)):
        trials = N
        temp_b = []
        temp_d = []
        while trials > 0:
            g = create_random_graph(nodes[i], edges[i], wmin, wmax)
            source = randint(0, nodes[i] - 1)
            mid = (nodes[i] - 1)//2
            k = randint(int(mid * 0.75), int(mid * 1.25))

            target = copy.deepcopy(g)
            temp = dijkstra_mem(target, source, k)
            temp_d.append(temp)

            target = copy.deepcopy(g)
            temp = bellmanFord_mem(target, source, k)
            temp_b.append(temp)
            trials -= 1
        d.append(np.mean(temp_d))
        bf.append(np.mean(temp_b))

    run_arr = [d, bf]
    xbar = nodes
    labels = ["Dijkstra", "BellmanFord"]
    means = [np.mean(d), np.mean(bf)]
    colors = ["blue", "orange"]
    # print(d, bf)
    color_plot_comparison_mem(run_arr, xbar, labels, "50_density_proper_k", means, colors, "nodes")
    return 0
# space cost density_test_fix_nodes=10_proper_k
def experiment9():
    N = 25
    nodes = 10
    edges = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    bf, d = [], []
    wmin, wmax = 1, 5
    for i in range(len(edges)):
        trials = N
        temp_b = []
        temp_d = []
        while trials > 0:
            g = create_random_graph(nodes, edges[i], wmin, wmax)
            source = randint(0, nodes - 1)
            mid = (nodes - 1)//2
            k = randint(int(mid * 0.75), int(mid * 1.25))

            target = copy.deepcopy(g)
            temp = dijkstra_mem(target, source, k)
            temp_d.append(temp)

            target = copy.deepcopy(g)
            temp = bellmanFord_mem(target, source, k)
            temp_b.append(temp)
            trials -= 1

        d.append(np.mean(temp_d))
        bf.append(np.mean(temp_b))

    run_arr = [d, bf]
    emax = nodes *( nodes - 1) // 2
    xbar = [int((e / emax)*100) for e in edges]
    labels = ["Dijkstra", "BellmanFord"]
    means = [np.mean(d), np.mean(bf)]
    colors = ["blue", "orange"]
    # print(d, bf)
    color_plot_comparison_mem(run_arr, xbar, labels, "density_test_fix_nodes=10_proper_k", means, colors, "%" + " density")
    return 0
experiment1()
experiment2()
experiment3()
experiment4()
experiment5()
experiment6()
experiment7()
experiment8()
experiment9()
