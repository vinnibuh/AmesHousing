import pytest
import housinglib.func as func
from housinglib.utils import csv_to_samples


@pytest.mark.parametrize('disable_grid_search',
                         [True, False])
def test_training_function(disable_grid_search):
    X_train, y_train = csv_to_samples('./data/processed/stable/train.csv')
    model = func.train(X_train, y_train,
                       disable_grid_search=disable_grid_search)

    assert model is not None


def test_testing_function():
    X_train, y_train = csv_to_samples('./data/processed/stable/train.csv')
    X_test, y_test = csv_to_samples('./data/processed/stable/test.csv')

    model = func.train(X_train, y_train, disable_grid_search=True)
    results = func.test(X_test, y_test, model)

    assert results['MSLE'] < 1
    assert results['R2'] > 0.7
