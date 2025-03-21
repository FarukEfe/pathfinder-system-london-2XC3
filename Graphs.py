from math import sqrt

# Using adjacency list implementation
class Graph:

    def __init__(self, nodes: int):
        self.graph: dict[int: list[int]] = {}
        self.weights: dict[tuple[int,int]: float] = {}
        for node in range(nodes):
            self.graph[node] = []

    def add_node(self, node: int):
        nodes = list(self.graph.keys())
        if node in nodes: return 
        self.graph[node] = []

    def has_edge(self, start: int, end: int):
        return self.graph[start] and (end in self.graph[start])

    def add_edge(self, start:int, end:int, w:float):
        nodes = list(self.graph.keys())
        if start not in nodes or end not in nodes:
            #print('Discarding `has_edge` call since not both endpoints exist.')
            return
        
        if self.has_edge(start, end):
            #print(f'Edge ({start},{end}) already exists. Updating weight...')
            self.weights[(start,end)] = w
            self.weights[(end,start)] = w

        self.graph[start].append(end)
        self.weights[(start,end)] = w

        self.graph[end].append(start)
        self.weights[(end,start)] = w

    def get_num_of_nodes(self) -> int:
        return len(list(self.graph.keys()))

    def w(node:int) -> float: # Why have this here but not the bottom? (Take it up w TA or Prof)
        pass

class WeightedGraph(Graph):

    def __init__(self, nodes):
        super().__init__(nodes=nodes)

    def w(node1: int, node2: int) -> float:
        pass

# Inquire more about what heuristics is
class HeuristicGraph(WeightedGraph):

    def __init__(self, nodes: int, dest: int, coords: tuple[float,float]):
        super().__init__(nodes)
        self.dest = dest
        self.x, self.y = coords[0], coords[1]
        self.__heuristic: dict[int,float] = []

    def get_heuristic(self) -> dict[int,float]:
        return self.__heuristic
    
    def set_heuristic(self, src: int, coords: tuple[float,float]):
        x, y = coords
        distance = sqrt(abs(self.x - x)**2 + abs(self.y - y)**2)
        self.__heuristic[src] = distance
        