from Graphs import *
from Algorithms import AStar, BellmanFord, Dijkstra, SPAlgorithm


class ShortPathFinder:

    def __init__(self):
        self.algorithm: SPAlgorithm = SPAlgorithm() # Default init
        self.graph: Graph = None
        self.heuristic_data: dict[int, tuple[float,float]] = None

    def calc_short_path(self, source:int, dest:int = None, k:int = None):
        if isinstance(self.algorithm, AStar):
            if not self.heuristic_data or not dest: return -1 # Cannot compute AStar without heuristic
            if dest not in self.heuristic_data.keys(): return -2 # Heuristic table doesn't have all nodes in station table
            return self.algorithm.calc_sp(self.graph, source, dest, self.heuristic_data)
        
        if not k: return -1
        return self.algorithm.calc_sp(self.graph, source, k)
    
    def set_graph(self, graph: Graph):
        self.graph = graph

    def set_algorithm(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm

    # Not in UML
    def set_heuristic(self, heuristic_data: dict[int, tuple[float,float]]):
        self.heuristic_data = heuristic_data