from DataLoader import DataLoader
from ShortPathFinder import ShortPathFinder
from Algorithms import AStar, BellmanFord, Dijkstra

if __name__ == "__main__":

    data = DataLoader('./Dataset')
    finder = ShortPathFinder()
    graph, heuristic = data.graph(), data.heuristic_data()

    # A star
    # finder.set_graph(graph)
    # finder.set_algorithm(AStar())
    # finder.set_heuristic(heuristic)
    
    # res = finder.calc_short_path(2, 8)
    # try:
    #     pred_dict, path = res
    #     print(pred_dict[1])
    #     print(path)
    # except:
    #     print(res)

    # BF
    finder.set_graph(graph)
    finder.set_algorithm(BellmanFord())
    #res = finder.calc_short_path(2, 8)
    #res = finder.calc_short_path(2, dest=8)
    distances, paths = finder.calc_short_path(2, k=15)
    print(distances)
    print(paths)

    # # D
    # finder.set_graph(graph)
    # finder.set_algorithm(Dijkstra())
    # #res = finder.calc_short_path(2, 8)
    # #res = finder.calc_short_path(2, dest=8)
    # distances = finder.calc_short_path(2, k=15)
    # print(distances)
    # # print(paths)