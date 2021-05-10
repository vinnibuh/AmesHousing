import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
from sys import path
path.append('.')
warnings.filterwarnings('ignore')
random_state = 17


def main():
    df_test = pd.read_csv(args.data_path)
    X_val, y_val = df_test.drop(['SalePrice'], axis=1), df_test.SalePrice
    regr = pickle.load(open(args.input_path, 'rb'))

    y_predict = regr.predict(X_val)
    r2_val = r2_score(y_val, y_predict)
    mse_val = mean_squared_error(y_val, y_predict)
    print('Validation MSLE value: ', mse_val)
    print('Validation R2 value: ', r2_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for testing")
    parser.add_argument('-i', '--input-path', type=str, default='./models/model.pk',
                        help='path to pretrained model file made by train.py')
    parser.add_argument('-d', '--data-path', type=str, default='./data/test.csv',
                        help='path to train data made by utils.split_dataset')
    args = parser.parse_args()
    main()
