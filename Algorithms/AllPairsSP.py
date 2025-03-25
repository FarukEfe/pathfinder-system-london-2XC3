from Algorithms.Dijkstra import Dijkstra
from Algorithms.BellmanFord import BellmanFord
from Graphs import WeightedGraph

class AllPairsSP:
    def __init__(self, use_bellman_ford=False, k=1):
        """
        :param use_bellman_ford: if True, use Bellman-Ford; else, use Dijkstra
        :param k: number of relaxation iterations (used in both algorithms)
        """
        self.use_bellman_ford_flag = use_bellman_ford
        self.k = k
        self.sp_algorithm = None  # Will initialize in calc_all_pairs_sp()

    def _has_negative_weights(self, graph: WeightedGraph):
        for (u, v), weight in graph.weights.items():
            if weight < 0:
                return True
        return False
    
    def calc_all_pairs_sp(self, graph: WeightedGraph):
        """
        Computes shortest path from every node to every other node.
        :return: (dist_table, prev_table)
                 where dist_table[u][v] is distance from u to v
                       prev_table[u][v] is the predecessor of v on the shortest path from u
        """
        if self.use_bellman_ford_flag is None:
            has_neg = self._has_negative_weights(graph)
            self.sp_algorithm = BellmanFord() if has_neg else Dijkstra()
        else:
            self.sp_algorithm = BellmanFord() if self.use_bellman_ford_flag else Dijkstra()

        dist_table = {}
        prev_table = {}

        for source in graph.graph.keys():
            dist, prev = self.sp_algorithm.calc_sp(graph, source, self.k)
            dist_table[source] = dist
            prev_table[source] = prev

        return dist_table, prev_table
