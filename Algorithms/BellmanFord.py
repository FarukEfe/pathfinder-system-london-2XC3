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
            # Copy of distances to avoid interference during updates
            new_distances = copy.deepcopy(distances)

            # For each edge in the graph
            for u in graph.graph.keys():
                for v in graph.graph[u]:
                    weight = graph.w(u, v)
                    if distances[u] + weight < new_distances[v]:
                        prev[v] = u
                        new_distances[v] = distances[u] + weight

            distances = new_distances

        return distances, prev