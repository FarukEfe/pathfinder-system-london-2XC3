from Algorithms.SPAlgorithm import SPAlgorithm
from Graphs import Graph, WeightedGraph
import heapq
import copy

class Dijkstra(SPAlgorithm):

    def __init__(self):
        pass

    def calc_sp(self, graph: WeightedGraph, source, k):
        # The Algorithm Comes Here
        # Initialize the priority queue and distances
        distances = {vertex: float('inf') for vertex,_ in graph.graph.items()}
        distances[source] = 0
        pq = [(0, source)]  # (distance, vertex)

        # Perform k steps of Dijkstra's algorithm
        for _ in range(k):
            if not pq:
                break

            # Get the node with the smallest distance
            current_dist, u = heapq.heappop(pq)

            if current_dist > distances[u]:
                continue

            # Relax edges of the current node
            for v, weight in graph.graph[u]:
                distance = current_dist + weight
                if distance < distances[v]:
                    distances[v] = distance
                    heapq.heappush(pq, (distance, v))

        return distances