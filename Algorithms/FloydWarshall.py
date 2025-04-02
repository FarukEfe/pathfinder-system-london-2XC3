from Algorithms.SPAlgorithm import SPAlgorithm
from Graphs import WeightedGraph

class FloydWarshall(SPAlgorithm):

    def __init__(self):
        pass

    def calc_apsp(self, graph: WeightedGraph):
        nodes = list(graph.graph.keys())
        n = len(nodes)

        node_to_index = {node: idx for idx, node in enumerate(nodes)}
        index_to_node = {idx: node for node, idx in node_to_index.items()}

        dist = [[float('inf')] * n for _ in range(n)]
        prev = [[None] * n for _ in range(n)]

        for i in range(n):
            dist[i][i] = 0
            prev[i][i] = nodes[i]

        for u in graph.graph:
            for v in graph.graph[u]:
                i, j = node_to_index[u], node_to_index[v]
                dist[i][j] = graph.w(u, v)
                prev[i][j] = u

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        prev[i][j] = prev[k][j]

        result_dist = {}
        result_prev = {}
        for i in range(n):
            for j in range(n):
                u = index_to_node[i]
                v = index_to_node[j]
                if u not in result_dist:
                    result_dist[u] = {}
                    result_prev[u] = {}
                result_dist[u][v] = dist[i][j]
                result_prev[u][v] = prev[i][j]

        return result_dist, result_prev
