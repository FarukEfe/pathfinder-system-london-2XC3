from Algorithms import AStar, Dijkstra, BellmanFord, SPAlgorithm
from ShortPathFinder import ShortPathFinder as SPF
from DataLoader import DataLoader
from Graphs import WeightedGraph

from itertools import product
from random import sample, uniform, randint
import timeit

import matplotlib.pyplot as plt
import numpy as np

# MARK: GENERATORS

def create_random_graph(nodes: int, edges: int, w_min: float, w_max: float) -> WeightedGraph:
    node_list = [n for n in range(nodes)]
    edge_list = list(product(node_list, repeat=2))
    edge_list = sample(edge_list, k=edges)
    graph = WeightedGraph(nodes)
    for u,v in edge_list:
        w = uniform(w_min, w_max)
        graph.add_edge(u,v,w)
    return node_list, graph

def create_heuristic_data(_nodes: list[int], lat_min: float, lat_max: float, long_min: float, long_max: float) -> dict[int: tuple[float,float]]:
    data: dict[int: tuple[float,float]] = {}
    for n in _nodes:
        latitude = uniform(lat_min, lat_max)
        longitude = uniform(long_min, long_max)
        data[n] = (latitude, longitude)
    return data

# MARK: PLOT

def plot(_data, fname: str):
    labels, runs = _data['labels'], _data['runs']
    fig, axis = plt.subplots(len(runs), 1, figsize=(12,12))
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92, wspace=0.2, hspace=0.5)
    for i in range(len(runs)):
        run, label = runs[i], labels[i]
        mean, x, _max = np.mean(run), np.arange(1,len(run)+1,1), max(run) * 1.20
        axis[i].bar(x,run, color='blue')
        axis[i].axhline(mean, color='blue', linestyle='--', label=f'average: {mean:.5f}')
        axis[i].set_xlabel('Iterations')
        axis[i].set_ylabel('Run time in ms')
        axis[i].set_title(label)
        axis[i].legend()
    plt.savefig(fname)

# MARK: EXPERIMENT

def experiment_random():
    N = 200
    w_min, w_max = 0, 5
    lat_min, lat_max, long_min, long_max = 50,53,0,2
    finder = SPF()
    time_dijkstra, time_astar = [], []
    for _ in range(N):
        n_node = randint(20, 50)
        n_edge = randint(20, 50)
        _nodes, graph = create_random_graph(n_node, n_edge, w_min, w_max)
        _data = create_heuristic_data(_nodes, lat_min, lat_max, long_min, long_max)
        source, dest = sample(_nodes, k=2)
        
        finder.set_graph(graph)
        finder.set_heuristic(_data)

        # DJ Test
        finder.set_algorithm(Dijkstra())

        start = timeit.default_timer()
        finder.calc_short_path(source, k=1)
        end = timeit.default_timer()

        time_dijkstra.append(end - start)

        # A* Test
        finder.set_algorithm(AStar())

        start = timeit.default_timer()
        finder.calc_short_path(source, dest)
        end = timeit.default_timer()

        time_astar.append(end - start)

    labels = ['Dijkstra\'s', 'A Star']
    runs = [time_dijkstra, time_astar]
    _data = { 'labels': labels, 'runs': runs }
    plot(_data, fname='Part5.jpg')

def experiment_london():

    N = 40
    time_dijkstra, time_astar = [], []

    data = DataLoader('./Dataset')
    finder = SPF()
    graph, heuristic = data.graph(), data.heuristic_data()
    finder.set_graph(graph)
    finder.set_heuristic(heuristic)
    _nodes = list(graph.graph.keys())
    for i in range(N):

        print(f'Iteration {i+1}')
        
        source, dest = sample(_nodes, k=2)
        
        # DJ Test
        finder.set_algorithm(Dijkstra())

        start = timeit.default_timer()
        finder.calc_short_path(source, k=1)
        end = timeit.default_timer()

        time_dijkstra.append(end - start)

        # A* Test
        finder.set_algorithm(AStar())

        start = timeit.default_timer()
        finder.calc_short_path(source, dest)
        end = timeit.default_timer()

        time_astar.append(end - start)

    labels = ['Dijkstra\'s', 'A Star']
    runs = [time_dijkstra, time_astar]
    _data = { 'labels': labels, 'runs': runs }
    
    plot(_data, fname='Part5.jpg')
    
experiment_london()
    