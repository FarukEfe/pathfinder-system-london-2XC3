from DataLoader import DataLoader
from ShortPathFinder import ShortPathFinder

if __name__ == "__main__":

    data = DataLoader('./.csv')
    finder = ShortPathFinder()

    finder.set_graph(data.graph())
    finder.calc_short_path(2, 8)