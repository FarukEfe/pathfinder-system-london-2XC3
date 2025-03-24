from Algorithms.SPAlgorithm import SPAlgorithm
from Graphs import HeuristicGraph, Graph, WeightedGraph

import heapq, time

class AStar(SPAlgorithm):

    def __init__(self):
        pass

    '''
    TO-DO'S:
    - 
    '''
    def calc_sp(self, graph: WeightedGraph, source: int, dest: int, heuristic: dict[int: tuple[float,float]]):
        # Base case return
        nodes = graph.graph.keys()
        if source not in nodes or dest not in nodes: return -1
        # Our heuristic is the latitude / longitude distance between the stations.
        # We actively pick the station that we can travel to in the shortest time, and has the smallest heuristic to destination
        heuristic_graph: HeuristicGraph = self.__compute_heuristic(dest, heuristic) # Get heuristic value from each node to destination
        h_table = heuristic_graph.get_heuristic() # Distance list of each node to the end

        came_from: dict[int:int] = { source: -1 } # Predecessor dictionary
        open_set = [source] # Initialize open list & predecessor list
        heapq.heapify(open_set)

        # cheapest path from start to n currently known
        g_score = { k:float('inf') for k,_ in graph.graph.items() }
        f_score = { k:float('inf') for k,_ in graph.graph.items() }
        g_score[source] = 0
        f_score[source] = h_table[source]

        while len(open_set) > 0:
            current = min(f_score, key=f_score.get) # node in open_set with lowest f_score value (maybe make more efficient?)

            if current == dest:
                path = self.reconstruct_path(came_from=came_from, current=current)
                came_from[source] = 0
                return (came_from, path)
        
            neighbors = graph.graph[current]
            for n in neighbors:
                tentative_g_score = g_score[current] + graph.w(current, n)
                if tentative_g_score < g_score[n]:
                    came_from[n] = current
                    g_score[n] = tentative_g_score
                    f_score[n] = tentative_g_score + h_table[n]
                    if n not in open_set:
                        heapq.heappush(open_set, n)
            # Remove current minimum since it cannot be revisited
            f_score.pop(current)
            open_set.remove(current)
        return -1

    def __compute_heuristic(self, dest: int, heuristic: dict[int: tuple[float,float]]) -> HeuristicGraph:
        # Compute heuristic graph from the latitude, longitude data
        nodes, coords = len(list(heuristic.keys())), heuristic[dest]
        heuristic_graph = HeuristicGraph(nodes, dest, coords)
        for k in list(heuristic.keys()):
            k_coords = heuristic[k]
            heuristic_graph.set_heuristic(k, k_coords)
        return heuristic_graph
    
    def reconstruct_path(self, came_from: list[int], current: int):
        n = current
        path = []
        while n != -1:
            path = [n] + path
            n = came_from[n]
        return path