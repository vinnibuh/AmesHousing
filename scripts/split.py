import argparse
import os
import datetime
import logging
from uuid import uuid4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from housinglib.cleansing import data_cleaning
from housinglib.eda import basic_feature_engineering


def main():
    logging.info('--- Splitting dataset ---')
    logging.info('Start Time: {}'.format(datetime.datetime.now()))
    logging.info('Arguments:')
    for k, v in vars(args).items():
        logging.info(f'{k}={v}')
    raw = pd.read_table(args.data_path, index_col=0)
    df = data_cleaning(raw)
    df = basic_feature_engineering(df)
    df['SalePrice'] = np.log(df['SalePrice'])

    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train.reset_index(drop=True)
    df_test.reset_index(drop=True)
    out_dir = os.path.join(args.processed_dir, args.run_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_train.to_csv(os.path.join(out_dir, 'train.csv'))
    df_test.to_csv(os.path.join(out_dir, 'test.csv'))
    logging.info('Data saved, location: {}'.format(out_dir))
    logging.info('Finish Time: {}'.format(datetime.datetime.now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for splitting and preprocessing AmesHousing dataset")
    parser.add_argument('-r', '--run-name', type=str, default='stable',
                        help='name of subdirectory to store results and logs')
    parser.add_argument('-d', '--data-path', type=str, default='./data/raw/AmesHousing.txt',
                        help='path to raw data from AmesHousing')
    parser.add_argument('-o', '--processed-dir', type=str, default='./data/processed',
                        help='subfolder to save preprocessed data')
    parser.add_argument('-l', '--log-path', type=str, default=None,
                        help='root directory of logs storage')
    args = parser.parse_args()
    if args.log_path is not None:
        log_dir = os.path.join(args.log_path, args.run_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, 'split_{}'.format(uuid4()))
        logging.basicConfig(filename=log_file, format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(message)s', level=logging.INFO)
    main()
