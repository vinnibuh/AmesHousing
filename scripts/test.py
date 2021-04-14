import argparse
import pickle
import warnings
from housinglib.utils import csv_to_samples
from housinglib import func

warnings.filterwarnings('ignore')
random_state = 17


def main():
    print('Testing model')
    X_val, y_val = csv_to_samples(args.data_path)
    model = pickle.load(open(args.input_path, 'rb'))
    results = func.test(X_val, y_val, model)
    print('Validation MSLE value: ', results['MSLE'])
    print('Validation R2 value: ', results['R2'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for testing")
    parser.add_argument('-i', '--input-path', type=str, default='./models/model.pk',
                        help='path to pretrained model file made by train.py')
    parser.add_argument('-d', '--data-path', type=str, default='./data/test.csv',
                        help='path to train data made by utils.split_dataset')
    args = parser.parse_args()
    main()
