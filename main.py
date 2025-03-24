from DataLoader import DataLoader
from ShortPathFinder import ShortPathFinder
from Algorithms import AStar, BellmanFord, Dijkstra

if __name__ == "__main__":

    data = DataLoader('./Dataset')
    finder = ShortPathFinder()
    graph, heuristic = data.graph(), data.heuristic_data()

    # A*
    # finder.set_graph(graph)
    # finder.set_algorithm(AStar())
    # finder.set_heuristic(heuristic)
    
    # res = finder.calc_short_path(2, 8)
    # try:
    #     pred_dict, path = res
    #     print(pred_dict)
    #     print(path)
    # except:
    #     print(res)

    # BELLMAN FORD
    # source, dest = 2, 8
    # finder.set_graph(graph)
    # finder.set_algorithm(BellmanFord())
    # distances, prev = finder.calc_short_path(source, k=15)
    # print(distances, prev)

    # DIJKSTRA
    source, dest = 2, 8
    finder.set_graph(graph)
    finder.set_algorithm(Dijkstra())
    dist, prev = finder.calc_short_path(source, k=15)
    print(prev, dist)

    # TEST
    # n = 8
    # path = []
    # while n != 2:
    #     path = [n] + path
    #     n = prev[n]
    # print(path)