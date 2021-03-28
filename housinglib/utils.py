"""Useful functions"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from housinglib.cleansing import data_cleaning
from housinglib.eda import basic_feature_engineering


def split_dataset(file_path='../data/AmesHousing.txt', output_directory='../data'):
    """
    Read raw dataset, preprocess (to some extent), split it on train/test and store in `csv` files.
    Names of subsets: `train.csv` and `test.csv`

    :param file_path: path to dataset file with extension `.txt`
    :param output_directory: path to folder, where data should be stored.
    :return:
    """
    raw = pd.read_table(file_path, index_col=0)
    df = data_cleaning(raw)
    df = basic_feature_engineering(df)
    df['SalePrice'] = np.log(df['SalePrice'])

    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train.reset_index(drop=True)
    df_test.reset_index(drop=True)
    df_train.to_csv(os.path.join(output_directory, 'train.csv'))
    df_test.to_csv(os.path.join(output_directory, 'test.csv'))
