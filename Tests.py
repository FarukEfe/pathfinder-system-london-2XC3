from Algorithms import AStar, Dijkstra, BellmanFord, SPAlgorithm
from ShortPathFinder import ShortPathFinder as SPF
from DataLoader import DataLoader
from Graphs import WeightedGraph

from itertools import product
from random import sample, uniform, randint
import timeit

import matplotlib.pyplot as plt
import numpy as np

# DJ Helper

def reconstruct_path(prev, n, p):
    # Find path
    path = []
    while n != p:
        path = [n] + path
        n = prev[n]
    return path

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

# MARK: GENERATOR

class Generator:

    def create_random_graph(self, nodes: int, edges: int, w_min: float, w_max: float) -> WeightedGraph:
        node_list = [n for n in range(nodes)]
        edge_list = list(product(node_list, repeat=2))
        edge_list = sample(edge_list, k=edges)
        graph = WeightedGraph(nodes)
        for u,v in edge_list:
            w = uniform(w_min, w_max)
            graph.add_edge(u,v,w)
        return node_list, graph

    def create_heuristic_data(self, _nodes: list[int], lat_min: float, lat_max: float, long_min: float, long_max: float) -> dict[int: tuple[float,float]]:
        data: dict[int: tuple[float,float]] = {}
        for n in _nodes:
            latitude = uniform(lat_min, lat_max)
            longitude = uniform(long_min, long_max)
            data[n] = (latitude, longitude)
        return data

# MARK: TESTS

class Tests:

    def test_random(self, n_node:int, n_edge:int, w_min:int=0, w_max:int=5, lat_min:float=50,lat_max:float=53,long_min:float=0,long_max:float=2):
        
        N, time_dijkstra, time_astar = 200, [], []
        finder, generator = SPF(), Generator()

        for _ in range(N):
            _nodes, graph = generator.create_random_graph(n_node, n_edge, w_min, w_max)
            _data = generator.create_heuristic_data(_nodes, lat_min, lat_max, long_min, long_max)
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
        _data = { 'labels': labels, 'runs': runs, 'hp': (n_node,n_edge) }
        plot(_data, fname='Part5.jpg')

    def test_london(self):

        N, time_dijkstra, time_astar = 300, [], []

        data, finder = DataLoader('./Dataset'), SPF()
        graph, heuristic = data.graph(), data.heuristic_data()
        finder.set_heuristic(heuristic)
        finder.set_graph(graph)

        _nodes = list(heuristic.keys())
        for _ in range(N):
            
            source, dest = sample(_nodes, k=2)
            
            # DJ Test
            finder.set_algorithm(Dijkstra())
            start = timeit.default_timer()
            finder.calc_short_path(source, k=5)
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
    
    def test_lines(self):
        # load .csv data
        data, finder = DataLoader('./Dataset'), SPF()
        graph, heuristic, lines = data.graph(), data.heuristic_data(), data.line_table()
        finder.set_heuristic(heuristic)
        finder.set_graph(graph)
        # sample a point
        src, dest = sample(list(heuristic.keys()), k=2)
        src, dest = 2,8
        # sp Dijkstra's
        finder.set_algorithm(Dijkstra())
        _, prev = finder.calc_short_path(src, k=1)
        p = reconstruct_path(prev, dest, src)
        n_line_dj = 0
        try:
            edges = [(p[i],p[i+1]) for i in range(len(p)-1)]
            _l = set([lines[edge] for edge in edges])
            n_line_dj = len(_l)
        except:
            pass
        print(f"Dijk: {n_line_dj}")
        # sp A*
        finder.set_algorithm(AStar())
        n_line_astar = 0
        try:
            _, p = finder.calc_short_path(src,dest)
            edges = [(p[i],p[i+1]) for i in range(len(p)-1)]
            _l = set([lines[edge] for edge in edges])
            n_line_astar = len(_l)
        except:
            pass
        print(f"A*: {n_line_astar}")

if __name__ == '__main__':
    tests = Tests()
    #tests.test_random(n_node=50,n_edge=500)
    tests.test_lines()

# TODO:
# - Write