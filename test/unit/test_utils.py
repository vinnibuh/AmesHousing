import pandas as pd
import housinglib.utils as utils


def test_train_csv_to_samples():
    data_path = './data/processed/stable/train.csv'
    X_train, y_train = utils.csv_to_samples(data_path)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert X_train.shape == (2264, 48)
    assert y_train.size == 2264

