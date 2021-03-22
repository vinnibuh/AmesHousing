from sys import path
path.append('..')
import argparse
import pandas as pd
from os import path
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from housinglib.eda import HousingTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from housinglib.utils import split_dataset
import joblib
import warnings
warnings.filterwarnings('ignore')
random_state = 17


def main():
    if not args.prepared_data:
        split_dataset(args.train_path, '../data')

    df_train = pd.read_csv('../data/train.csv')
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

    filename = args.model_name + '.pickle'
    out_path = path.join(args.output_dir, filename)
    if args.disable_grid_search:
        joblib.dump(regr.best_estimator_, out_path)
    else:
        joblib.dump(regr, out_path)
    print('Saved model at ', out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training resnet-v2")
    parser.add_argument('--output-dir', type=str, default='../models/', help='the directory to save the trained model')
    parser.add_argument('--train-path', type=str, default='../data/AmesHousing.txt', help='the input raw data file')
    parser.add_argument('--model-name', type=str, default='model', help='name of file to save')
    parser.add_argument('--prepared-data', action='store_true', default=False,
                        help='if input is one of the files generated from running the script split_data.py')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha in Ridge regressor')
    parser.add_argument('--n_components', type=int, default=6, choices=[1, 2, 3, 4, 5, 6],
                        help='number of PCA components to leave')
    parser.add_argument('--disable-grid-search', action='store_false', default=True,
                        help='true means using memonger to save momory, https://github.com/dmlc/mxnet-memonger')
    args = parser.parse_args()
    main()
