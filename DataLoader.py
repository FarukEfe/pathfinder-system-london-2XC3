import pandas as pd
import os

from Graphs import *

class DataLoader:
    
    def __init__(self, path:str):
        self.connections = pd.read_csv(os.path.join(path, 'london_connections.csv'))
        self.stations = pd.read_csv(os.path.join(path, 'london_stations.csv'))
    
    def graph(self) -> Graph:
        n_node = max(self.stations['id'])
        _connection: pd.DataFrame = self.connections[['station1', 'station2', 'time']]
        obj = WeightedGraph(n_node)
        _connection.itertuples()
        
        for i in range(len(_connection)):
            row = _connection.iloc[i]
            p, q, w = row['station1'], row['station2'], row['time']
            obj.add_edge(p, q, w)
        
        return obj


        
