import pandas as pd


def csv_to_samples(data_path):
    df = pd.read_csv(data_path)
    X, y = df.drop(['SalePrice'], axis=1), df.SalePrice
    return X, y
