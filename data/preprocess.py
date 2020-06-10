import pandas as pd
def process(config):
    return pd.read_csv(**config).dropna()
    
