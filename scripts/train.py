import argparse
import pandas as pd
import os
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pickle
import warnings
from sys import path
path.append('..')

from housinglib.eda import HousingTransformer

warnings.filterwarnings('ignore')
random_state = 17


def main():
    df_train = pd.read_csv(args.data_path)
    X_train, y_train = df_train.drop(['SalePrice'], axis=1), df_train.SalePrice
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
    estimators = [('transform', HousingTransformer()),
                  ('std', StandardScaler()),
                  ('model', Ridge())]
    regr = Pipeline(estimators)
    if args.disable_grid_search:
        param_grid = dict(model__alpha=[1e0, 1e-1, 1e-2],
                          transform__n_pca_components=[3, 4, 5, 6])
        regr = GridSearchCV(regr, scoring=['r2', 'neg_root_mean_squared_error'],
                            param_grid=param_grid, cv=rkf, refit='neg_root_mean_squared_error')
    else:
        regr.set_params(model__alpha=args.alpha,
                        transform__n_pca_components=args.n_components)
    regr.fit(X_train, y_train)

    out_path = args.output_path
    out_folder = os.path.dirname(out_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if args.disable_grid_search:
        pickle.dump(regr.best_estimator_, open(out_path, 'wb'))
    else:
        pickle.dump(regr, open(out_path, 'wb'))
    print('Saved model at ', out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training model on AmesHousing dataset")
    parser.add_argument('-d', '--data-path', type=str, default='../data/train.csv',
                        help='path to train data made by utils.split_dataset')
    parser.add_argument('-o', '--output-path', type=str, default='../models/model.pk',
                        help='path to save the model')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='alpha in Ridge regressor')
    parser.add_argument('-c', '--n_components', type=int, default=4, choices=[1, 2, 3, 4, 5, 6],
                        help='number of PCA components to leave')
    parser.add_argument('--disable-grid-search', action='store_false', default=True,
                        help='do not use GridSearch in training, use fixed model parameters')
    args = parser.parse_args()
    main()
