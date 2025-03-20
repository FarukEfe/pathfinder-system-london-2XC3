from Graphs import *
from Algorithms import SPAlgorithm

class ShortPathFinder:

    def __init__(self):
        self.algorithm: SPAlgorithm = SPAlgorithm() # Default init
        self.graph: Graph = None

    def calc_short_path(self, source:int, dest:int) -> float:
        return self.algorithm.calc_sp(self.graph, source, dest)

    def set_graph(self, graph: Graph):
        self.graph = graph

    def set_algorithm(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm