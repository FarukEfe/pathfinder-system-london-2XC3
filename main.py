from DataLoader import DataLoader
from ShortPathFinder import ShortPathFinder
from Algorithms import AStar, BellmanFord, Dijkstra

if __name__ == "__main__":

    data = DataLoader('./Dataset')
    finder = ShortPathFinder()
    graph, heuristic = data.graph(), data.heuristic_data()

    finder.set_graph(graph)
    finder.set_algorithm(AStar())
    finder.set_heuristic(heuristic)
    
    res = finder.calc_short_path(2, 8)
    try:
        pred_dict, path = res
        print(pred_dict)
        print(path)
    except:
        print(res)