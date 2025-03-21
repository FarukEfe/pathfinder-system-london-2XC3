from DataLoader import DataLoader
from ShortPathFinder import ShortPathFinder

if __name__ == "__main__":

    data = DataLoader('./.csv')
    finder = ShortPathFinder()
    graph, heuristic = data.graph(), data.heuristic_data()

    finder.set_graph(graph)
    finder.set_heuristic(heuristic)
    
    finder.calc_short_path(2, 8)