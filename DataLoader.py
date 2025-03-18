import pandas as pd

class DataLoader:
    
    def __init__(self, path:str):
        table = pd.read_csv(path)
        
