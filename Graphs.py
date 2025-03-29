from math import sqrt

# Using adjacency list implementation
class Graph:

    def __init__(self, nodes: int):
        self.graph: dict[int: list[int]] = {}
        self.weights: dict[tuple[int,int]: float] = {}
        for node in range(nodes):
            self.graph[node] = []
    
    def get_adj_nodes(self, node: int) -> list[int]:
        if node not in self.graph.keys(): return -1
        return self.graph[node]

    def add_node(self, node: int):
        nodes = list(self.graph.keys())
        if node in nodes: return 
        self.graph[node] = []

    # Not in UML
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
    
    def w(node: int) -> float:
        raise NotImplementedError(
            "Proper implementation is listed in WeightedGraph class type. Use that instance instead."
        )

class WeightedGraph(Graph):

    def __init__(self, nodes):
        super().__init__(nodes=nodes)

    def w(self, node1: int, node2: int) -> float:
        return self.weights[(node1,node2)]

class HeuristicGraph(WeightedGraph):

    def __init__(self, nodes: int, dest: int, coords: tuple[float,float]):
        super().__init__(nodes)
        self.dest = dest
        self.x, self.y = coords[0], coords[1]
        self.__heuristic: dict[int:float] = {}

    def get_heuristic(self) -> dict[int:float]:
        return self.__heuristic
    
    # Not in UML
    def set_heuristic(self, src: int, coords: tuple[float,float]):
        x, y = coords
        distance = sqrt(abs(self.x - x)**2 + abs(self.y - y)**2)
        self.__heuristic[src] = distance
        