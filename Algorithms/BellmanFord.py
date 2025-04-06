from Algorithms.SPAlgorithm import SPAlgorithm
from Graphs import Graph, WeightedGraph
import copy

class BellmanFord(SPAlgorithm):

    def __init__(self):
        pass

    def calc_sp(self, graph: WeightedGraph, source: int, k: int):
        # The Algorithm Comes Here
        # Initialize the distance to all vertices as infinity
        distances = {vertex: float('inf') for vertex,_ in graph.graph.items()}
        prev = {vertex: -1 for vertex,_ in graph.graph.items()}
        distances[source] = 0

        # Relax all edges up to k times
        for _ in range(k):

            # For each edge in the graph
            for u in graph.graph.keys():
                for v in graph.graph[u]:
                    weight = graph.w(u, v)
                    if distances[u] + weight < distances[v]:
                        prev[v] = u
                        distances[v] = distances[u] + weight

        return distances, prev