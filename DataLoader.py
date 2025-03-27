import pandas as pd
import os

from Graphs import *

class DataLoader:
    
    def __init__(self, path:str):
        self.connections = pd.read_csv(os.path.join(path, 'london_connections.csv'))
        self.stations = pd.read_csv(os.path.join(path, 'london_stations.csv'))
    
    def graph(self) -> Graph:
        n_node = max(self.stations['id']) + 1
        _connection: pd.DataFrame = self.connections[['station1', 'station2', 'time']]
        obj = WeightedGraph(n_node)
        
        for i in range(len(_connection)):
            row = _connection.iloc[i]
            unpack = lambda x: int(x)
            p, q, w = row['station1'], row['station2'], row['time']
            p, q, w = unpack(p), unpack(q), unpack(w)
            obj.add_edge(p, q, w)
        
        return obj

    def heuristic_data(self) -> dict[int,tuple[float,float]]:
        _data: pd.DataFrame = self.stations[['id', 'latitude', 'longitude']]
        obj: dict[int, tuple[float,float]] = {}
        for i in range(len(_data)):
            row = _data.iloc[i]
            id, x, y = int(row['id']), float(row['latitude']), float(row['longitude'])
            obj[id] = (x, y)
        return obj

    def line_table(self) -> dict[(int,int): int]:
        _data: pd.DataFrame = self.connections[['station1','station2','line']]
        obj: dict[(int,int): int] = {}
        for i in range(len(_data)):
            row = _data.iloc[i]
            unpack = lambda x: int(x)
            p, q, l = row['station1'], row['station2'], row['line']
            p, q, l = unpack(p), unpack(q), unpack(l)
            obj[(p,q)] = l
        return obj

        
