import pandas as pd

def load_business_data(path="data/business_data.csv"):
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    return df
