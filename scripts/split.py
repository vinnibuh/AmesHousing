import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sys import path
path.append('.')

from housinglib.cleansing import data_cleaning
from housinglib.eda import basic_feature_engineering


def main():
    """
    Read raw dataset, preprocess (to some extent), split it on train/test and store in `csv` files.
    Names of subsets: `train.csv` and `test.csv`

    :param file_path: path to dataset file with extension `.txt`
    :param output_directory: path to folder, where data should be stored.
    :return:
    """
    raw = pd.read_table(args.data_path, index_col=0)
    df = data_cleaning(raw)
    df = basic_feature_engineering(df)
    df['SalePrice'] = np.log(df['SalePrice'])

    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train.reset_index(drop=True)
    df_test.reset_index(drop=True)
    df_train.to_csv(os.path.join(args.output_path, 'train.csv'))
    df_test.to_csv(os.path.join(args.output_path, 'test.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for splitting and preprocessing AmesHousing dataset")
    parser.add_argument('-p', '--data-path', type=str, default='./data/AmesHousing.txt',
                        help='path to raw data from AmesHousing')
    parser.add_argument('-o', '--output-path', type=str, default='./data',
                        help='path to save preprocesses models')
    args = parser.parse_args()
    main()
