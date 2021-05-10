import warnings
import logging
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from housinglib.eda import HousingTransformer

warnings.filterwarnings('ignore')
random_state = 17


def train(X, y, alpha=1.0, n_components=4,
          disable_grid_search=False):
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random_state)
    estimators = [('transform', HousingTransformer()),
                  ('std', StandardScaler()),
                  ('model', Ridge())]
    regr = Pipeline(estimators)
    if not disable_grid_search:
        param_grid = dict(model__alpha=[1e0, 1e-1, 1e-2],
                          transform__n_pca_components=[3, 4, 5, 6])
        regr = GridSearchCV(regr, scoring=['r2', 'neg_root_mean_squared_error'],
                            param_grid=param_grid, cv=rkf, refit='neg_root_mean_squared_error')
    else:
        regr.set_params(model__alpha=alpha,
                        transform__n_pca_components=n_components)
    regr.fit(X, y)
    if not disable_grid_search:
        rmse_cv_score = regr.best_score_
        best_index = regr.best_index_
        r2_cv_score = regr.cv_results_['mean_test_r2'][best_index]
        logging.info('RMSLE CV score on train: {:.4f}'.format(rmse_cv_score))
        logging.info('R2 CV score on train: {:.4f}'.format(r2_cv_score))

        mse_train_score = mean_squared_error(y, regr.best_estimator_.predict(X))
        r2_train_score = r2_score(y, regr.best_estimator_.predict(X))
        logging.info('MSLE score on train: {:.4f}'.format(mse_train_score))
        logging.info('R2 score on train: {:.4f}'.format(r2_train_score))

        model = regr.best_estimator_
    else:
        mse_train_score = mean_squared_error(y, regr.predict(X))
        logging.info('MSLE score on train: {:.4f}'.format(mse_train_score))
        logging.info('R2 score on train: {:.4f}'.format(regr.score(X, y)))
        model = regr

    return model


def test(X, y, model):
    y_predict = model.predict(X)
    r2_val = r2_score(y, y_predict)
    mse_val = mean_squared_error(y, y_predict)
    logging.info('MSLE score on test: {:.4f}'.format(mse_val))
    logging.info('R2 score on test: {:.4f}'.format(r2_val))
    return {'MSLE': mse_val,
            'R2': r2_val,
            'y_predict': y_predict,
            'y_true': y,
            'X': X}
