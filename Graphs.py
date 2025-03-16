class Graph:

    def __init__(self):
        pass

    def add_node(node: int):
        pass

    def add_edge(start:int, end:int, w:float):
        pass

    def get_num_of_nodes() -> int:
        pass

    def w(node:int) -> float:
        pass

class WeightedGraph(Graph):

    def w(node1: int, node2: int) -> float:
        pass

class HeuristicGraph(WeightedGraph):

    def __init__(self):
        super().__init__()
        self.__heuristic: dict[int,float]

    def get_heuristic(self) -> dict[int,float]:
        return self.__heuristic