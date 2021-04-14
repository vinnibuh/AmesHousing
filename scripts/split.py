import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from housinglib.cleansing import data_cleaning
from housinglib.eda import basic_feature_engineering


def main():
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
