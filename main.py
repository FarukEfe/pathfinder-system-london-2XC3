from DataLoader import DataLoader
from ShortPathFinder import ShortPathFinder
from Algorithms import AStar, BellmanFord, Dijkstra

if __name__ == "__main__":

    data = DataLoader('./.csv')
    finder = ShortPathFinder()
    graph, heuristic = data.graph(), data.heuristic_data()

    finder.set_graph(graph)
    finder.set_algorithm(AStar())
    finder.set_heuristic(heuristic)

    finder.calc_short_path(2, 8)