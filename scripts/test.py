import argparse
import pickle
import warnings
import os
import datetime
import logging
from uuid import uuid4
from housinglib.utils import csv_to_samples
from housinglib import func

warnings.filterwarnings('ignore')
random_state = 17


def main():
    logging.info('--- Testing Model ---')
    logging.info('Start Time: {}'.format(datetime.datetime.now()))
    logging.info('Arguments:')
    for k, v in vars(args).items():
        logging.info(f'{k}={v}')
    X_val, y_val = csv_to_samples(args.data_path)
    model_path = os.path.join(args.models_path, args.run_name, 'model.pk')
    model = pickle.load(open(model_path, 'rb'))
    results = func.test(X_val, y_val, model)
    out_dir = os.path.join(args.results_path, args.run_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, 'results.pk')
    pickle.dump(results, open(out_path, 'wb'))
    logging.info('Finish Time: {}'.format(datetime.datetime.now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for testing")
    parser.add_argument('-r', '--run-name', type=str, default='stable',
                        help='name of subdirectory to store results and logs')
    parser.add_argument('-m', '--models-path', type=str, default='./models',
                        help='root directory of models storage')
    parser.add_argument('-d', '--data-path', type=str, default='./data/processed/stable/test.csv',
                        help='path to train data file made by utils.split_dataset')
    parser.add_argument('-R', '--results-path', type=str, default='./results',
                        help='root folder of results storage')
    parser.add_argument('-l', '--log-path', type=str, default=None,
                        help='root directory of logs storage')
    args = parser.parse_args()
    if args.log_path is not None:
        log_dir = os.path.join(args.log_path, args.run_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, 'test_{}'.format(uuid4()))
        logging.basicConfig(filename=log_file, format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(message)s', level=logging.INFO)
    main()
