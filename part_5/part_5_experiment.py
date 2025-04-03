import sys, os
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

def scatter(_data, fname: str):
    astar, dijkstra, edges = _data['runs_astar'], _data['runs_dijkstra'], _data['edges']
    fig, ax = plt.figure(figsize=(12,12)), plt.gca()
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92, wspace=0.2, hspace=0.5)
    plt.plot(edges, dijkstra, color='blue', label="Dijkstra's")
    plt.plot(edges, astar, color='orange', label="A*")
    plt.title('Runtime: Dijkstra\'s vs. A* by Graph Density', fontsize=24)
    plt.ylabel('Run time in ms', fontsize=20)
    plt.xlabel('# Edges', fontsize=20)
    plt.grid(color='grey', linewidth=1)
    ax.set_axisbelow(True)
    plt.legend(loc=2, prop={'size': 20})
    plt.savefig(fname)

def plot2(_data, fname: str):
    astars, djs, lab = _data['astars'], _data['djs'], _data['labels']
    fig, ax = plt.figure(figsize=(12,12)), plt.gca()
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92, wspace=0.2, hspace=0.5)
    bar1=np.arange(len(lab))
    bar2=[i+0.45 for i in bar1]
    plt.bar(bar1,djs,0.4,color='mediumaquamarine',label='Dijkstra\'s')
    plt.bar(bar2,astars,0.4,color='blue',label='A*')
    plt.title('Runtime Comparison by Line Transfers', fontsize=24)
    plt.ylabel('Run time in ms', fontsize=20)
    plt.xlabel('# Transfers', fontsize=20)
    plt.xticks(np.add(bar1,bar2)/2,lab, fontsize=16)
    plt.grid(color='grey', linewidth=1)
    ax.set_axisbelow(True)
    plt.legend(loc=2, prop={'size': 20})
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

    def test_random(self, n_node:int, n_edge:int, 
                    w_min:int=0, w_max:int=5,
                    lat_min:float=50, lat_max:float=53, 
                    long_min:float=0,long_max:float=2,
                    file_name: str = 'P5.jpg'
                    ):
        
        N, time_dijkstra, time_astar = 200, [], []
        finder, generator = SPF(), Generator()

        for i in range(N):

            print(f'Test Density ({i})', end='\r')

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
        plot(_data, fname=file_name)

        avg_astar, avg_dijkstra = sum(time_astar)/len(time_astar), sum(time_dijkstra)/len(time_dijkstra)
        return avg_astar, avg_dijkstra
    
    def test_london(self, points: list[tuple[int,int]] = None, file_name: str = 'P5.jpg'):

        N, time_dijkstra, time_astar = 300, [], []
        if points is not None: N = len(points)

        data, finder = DataLoader('./Dataset'), SPF()
        graph, heuristic = data.graph(), data.heuristic_data()
        finder.set_heuristic(heuristic)
        finder.set_graph(graph)

        _nodes = list(heuristic.keys())
        for i in range(N):

            print(f'Test London ({i})', end='\r')

            source, dest = sample(_nodes, k=2)
            if points is not None: source, dest = points[i]

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
        plot(_data, fname=file_name)
        
        avg_astar, avg_dijkstra = sum(time_astar)/len(time_astar), sum(time_dijkstra)/len(time_dijkstra)
        return avg_astar, avg_dijkstra
    
    def test_lines(self):
        # load .csv data
        data, finder = DataLoader('./Dataset'), SPF()
        graph, heuristic, lines = data.graph(), data.heuristic_data(), data.line_table()
        finder.set_heuristic(heuristic)
        finder.set_graph(graph)
        # sample a point
        src, dest = sample(list(heuristic.keys()), k=2)
        # sp Dijkstra's
        finder.set_algorithm(Dijkstra())
        _, prev = finder.calc_short_path(src, k=1)
        p = reconstruct_path(prev, dest, src)
        n_line_dj = 0
        edges = [(p[i],p[i+1]) for i in range(len(p)-1)]
        _l = set([(lines[edge] if edge in lines.keys() else lines[(edge[1], edge[0])]) for edge in edges])
        n_line_dj = len(_l)
        #print(f"Dijk: {n_line_dj}")

        # sp A*
        finder.set_algorithm(AStar())
        n_line_astar = 0
        try:
            _, p = finder.calc_short_path(src,dest)
            edges = [(p[i],p[i+1]) for i in range(len(p)-1)]
            _l = set([(lines[edge] if edge in lines.keys() else lines[(edge[1], edge[0])]) for edge in edges])
            n_line_astar = len(_l)
        except:
            pass
        #print(f"A*: {n_line_astar}")
        # Return the minimum line switches between src and dest
        return (src, dest, min(n_line_dj, n_line_astar))

if __name__ == '__main__':
    # Create plot folder if doesn't exist
    if not os.path.exists(os.path.join(os.getcwd(),'plots')): os.mkdir(os.path.join(os.getcwd(),'plots'))
    if not os.path.exists(os.path.join(os.path.join(os.getcwd(),'plots'), 'bars')): os.mkdir(os.path.join(os.path.join(os.getcwd(),'plots'),'bars'))
    tests = Tests()
    # Test A* vs Dijkstra's on London Dataset
    print('\nStep One\n')
    _ = tests.test_london(file_name='plots/P5_London.jpg')

    # Test A* vs Dijkstra's on Varying Densities
    print('\nStep Two\n')
    n_node, n_edges = 100, list(np.concatenate(np.array([[(10**i)*j for j in range(1,11)] for i in range(1,4)])))
    astar_runs, dijkstra_runs = [], []
    for edge in n_edges:
        print(f'\nEdge: {edge}\n')
        t_astar, t_dijkstra = tests.test_random(n_node=n_node,n_edge=edge,file_name=f'plots/bars/P5_Edge_{edge}.jpg')
        astar_runs.append(t_astar)
        dijkstra_runs.append(t_dijkstra)
    
    density_run_data = {
        'runs_astar': astar_runs,
        'runs_dijkstra': dijkstra_runs,
        'edges': n_edges
    }

    scatter(density_run_data, fname='plots/astar_dijkstra_density_runtimes.jpg')

    # Get line switches of optimal path for varying points and classify them
    line_switch = {
        'same_line': [],
        'one_line': [],
        'multiple_line': []
    }

    print('\nStep Three\n')
    for i in range(1000):
        print(f'Test Lines ({i})', end='\r')
        p, q, n = tests.test_lines()
        if n == 1: line_switch['same_line'].append((p,q))
        if n == 2: line_switch['one_line'].append((p,q))
        else: line_switch['multiple_line'].append((p,q))
    
    # # Make sure all lists have same length for fair comparison
    min_len = min(len(line_switch['same_line']), len(line_switch['one_line']), len(line_switch['multiple_line']))
    line_switch['same_line'], line_switch['one_line'], line_switch['multiple_line'] = line_switch['same_line'][:min_len], line_switch['one_line'][:min_len], line_switch['multiple_line'][:min_len]
    
    # # Test A* vs Dijkstra's on varying line switches in optimal path
    print('\nStep Four\n')
    same_avg_astar, same_avg_dj = tests.test_london(line_switch['same_line'], file_name='plots/bars/P5_0_Line.jpg')
    one_avg_astar, one_avg_dj = tests.test_london(line_switch['one_line'], file_name='plots/bars/P5_1_Line.jpg')
    mult_avg_astar, mult_avg_dj = tests.test_london(line_switch['multiple_line'], file_name='plots/bars/P5_2_Line.jpg')
    astars = [same_avg_astar, one_avg_astar, mult_avg_astar]
    djs = [same_avg_dj, one_avg_dj, mult_avg_dj]
    labels = ['Same Line', 'One Line', 'Mult. Lines']
    line_data = {
        'djs': djs,
        'astars': astars,
        'labels': labels
    }
    plot2(line_data, fname='plots/compare_astar_dj_lines.jpg')