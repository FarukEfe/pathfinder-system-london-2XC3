from Algorithms.SPAlgorithm import SPAlgorithm
from Graphs import Graph, WeightedGraph
import heapq
from copy import deepcopy, copy

class Dijkstra(SPAlgorithm):

    def __init__(self):
        pass

    def calc_sp(self, graph: WeightedGraph, source: int, k: int):

        dist_table = { v: float('inf') for v in graph.graph.keys() }
        prev_table = { v: -1 for v in graph.graph.keys() }
        dist_table[source], prev_table[source] = 0, source
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


