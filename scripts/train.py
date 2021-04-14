import warnings
import argparse
import os
import pickle
from housinglib.utils import csv_to_samples
from housinglib import func

warnings.filterwarnings('ignore')
random_state = 17


def main():
    print('Training model...')
    X_train, y_train = csv_to_samples(args.data_path)
    model_ = func.train(X_train, y_train, args.alpha,
                        args.n_components, args.disable_grid_search)
    print('Training is successful')

    out_path = args.output_path
    out_folder = os.path.dirname(out_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    pickle.dump(model_, open(out_path, 'wb'))
    print('Saved model at ', out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training model on AmesHousing dataset")
    parser.add_argument('-d', '--data-path', type=str, default='./data/train.csv',
                        help='path to train data made by scripts/split.py')
    parser.add_argument('-o', '--output-path', type=str, default='./models/model.pk',
                        help='path to save the model')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha in Ridge regressor')
    parser.add_argument('-c', '--n_components', type=int, default=4, choices=[1, 2, 3, 4, 5, 6],
                        help='number of PCA components to leave')
    parser.add_argument('--disable-grid-search', action='store_true', default=False,
                        help='do not use GridSearch in training, use fixed model parameters')
    args = parser.parse_args()
    main()
