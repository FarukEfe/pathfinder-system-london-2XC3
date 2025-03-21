from Graphs import *
from Algorithms import SPAlgorithm, AStar

class ShortPathFinder:

    def __init__(self):
        self.algorithm: SPAlgorithm = SPAlgorithm() # Default init
        self.graph: Graph = None
        self.heuristic_data: dict[int, tuple[float,float]] = None

    def calc_short_path(self, source:int, dest:int) -> float:
        if isinstance(self.algorithm, AStar):
            if not self.heuristic_data: return None # Cannot compute AStart without heuristic
            return self.algorithm.calc_sp(self.graph, source, dest, self.heuristic_data)
        return self.algorithm.calc_sp(self.graph, source, dest)

    def set_graph(self, graph: Graph):
        self.graph = graph
    
    def set_heuristic(self, heuristic_data: dict[int, tuple[float,float]]):
        self.heuristic_data = heuristic_data

    def set_algorithm(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm