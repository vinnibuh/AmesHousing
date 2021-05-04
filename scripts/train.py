import warnings
import argparse
import os
import pickle
import logging
import datetime
from uuid import uuid4
from housinglib.utils import csv_to_samples
from housinglib import func

warnings.filterwarnings('ignore')
random_state = 17


def main():
    logging.info('--- Training Model ---')
    logging.info('Start Time: {}'.format(datetime.datetime.now()))
    logging.info('Arguments:')
    for k, v in vars(args).items():
        logging.info(f'{k}={v}')
    X_train, y_train = csv_to_samples(args.data_path)
    model_ = func.train(X_train, y_train, args.alpha,
                        args.n_components, args.disable_grid_search)
    logging.info('Training is finished')
    logging.info('Time: {}'.format(datetime.datetime.now()))

    out_folder = os.path.join(args.models_path, args.run_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_path = os.path.join(out_folder, 'model.pk')
    pickle.dump(model_, open(out_path, 'wb'))
    logging.info('Models saved, location: {}'.format(out_path))
    logging.info('Finish Time: {}'.format(datetime.datetime.now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training model on AmesHousing dataset")
    parser.add_argument('-r', '--run-name', type=str, default='stable',
                        help='name of subdirectory to store results and logs')
    parser.add_argument('-d', '--data-path', type=str, default='./data/processed/stable/train.csv',
                        help='path to train data file made by scripts/split.py')
    parser.add_argument('-o', '--models-path', type=str, default='./models',
                        help='root directory of models storage')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha in Ridge regressor')
    parser.add_argument('-c', '--n_components', type=int, default=4, choices=[1, 2, 3, 4, 5, 6],
                        help='number of PCA components to leave')
    parser.add_argument('--disable-grid-search', action='store_true', default=False,
                        help='do not use GridSearch in training, use fixed model parameters')
    parser.add_argument('-l', '--log-path', type=str, default=None,
                        help='root directory of logs storage')
    args = parser.parse_args()
    if args.log_path is not None:
        log_dir = os.path.join(args.log_path, args.run_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, 'train_{}'.format(uuid4()))
        logging.basicConfig(filename=log_file, format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(message)s', level=logging.INFO)
    main()
