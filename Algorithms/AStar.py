from SPAlgorithm import SPAlgorithm
from Graphs import HeuristicGraph

class AStar(SPAlgorithm):

    def __init__(self):
        pass

    def calc_sp(self, graph, source, dest, heuristic: dict[int, tuple[float,float]]):
       # Our heuristic is the latitude / longitude distance between the stations.
       # We actively pick the station that we can travel to in the shortest time, and has the smallest heuristic to destination
       heuristic_graph = self.__compute_heuristic(heuristic) # Get heuristic value from each node to destination
       return

    def __compute_heuristic(self, dest, heuristic: dict[int, tuple[float,float]]) -> HeuristicGraph:
        # Compute heuristic graph from the latitude, longitude data
        nodes, coords = len(list(heuristic.keys())), heuristic[dest]
        heuristic_graph = HeuristicGraph(nodes, dest, coords)
        for k in list(heuristic.keys()):
            k_coords = heuristic[k]
            heuristic_graph.set_heuristic(k, k_coords)
        return heuristic_graph