import pandas as pd


def csv_to_samples(data_path):
    df_train = pd.read_csv(data_path)
    X_train, y_train = df_train.drop(['SalePrice'], axis=1), df_train.SalePrice
    return X_train, y_train
