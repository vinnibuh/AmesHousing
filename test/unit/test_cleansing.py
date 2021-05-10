import pytest
import numpy as np
import pandas as pd
from sys import path

path.append('.')

from housinglib.cleansing import drop_precision, fill_na_real, data_cleaning


def test_drop_precision():
    """Checks if drop_precision function works correctly: drops precision,
    keeps shape intact and doesn't change values."""
    rng = np.random.default_rng()
    test_floats = pd.DataFrame(rng.random((100, 10), dtype='float64'),
                               columns=['f_' + str(n) for n in range(10)])
    test_ints = pd.DataFrame(rng.integers(0, 10, (100, 10), dtype='int64'),
                             columns=['i_' + str(n) for n in range(10)])
    test_df = pd.concat([test_ints, test_floats], axis=1)
    result = drop_precision(test_df)

    assert result.shape == test_df.shape

    old_int_cols = test_df.select_dtypes(include='int64').columns
    new_int_cols = result.select_dtypes(include='int32').columns
    assert (old_int_cols == new_int_cols).all()

    old_float_cols = test_df.select_dtypes(include='float64').columns
    new_float_cols = result.select_dtypes(include='float32').columns
    assert (old_float_cols == new_float_cols).all()

    assert np.allclose(test_df.values, result.values)


def test_fill_na_real():
    """Checks if fill_na_real function works correctly: fills all NaNs, doesn't change shape
     and fills with correct values."""
    rng = np.random.default_rng()
    test_floats = pd.DataFrame(rng.random((100, 10), dtype='float64'),
                               columns=['f_' + str(n) for n in range(10)])
    test_ints = pd.DataFrame(rng.integers(0, 10, (100, 10), dtype='int32'),
                             columns=['i_' + str(n) for n in range(10)])
    test_df = pd.concat([test_floats, test_ints], axis=1)

    nan_idx_x = rng.choice(100, size=250)
    nan_idx_y = rng.choice(20, size=250)

    for i in range(250):
        test_df.iloc[nan_idx_x[i], nan_idx_y[i]] = np.NaN

    fill_values = pd.concat([test_df.iloc[:, :10].mean(),
                             test_df.iloc[:, 10:].median()], axis=0)
    result = fill_na_real(test_df)

    assert result.shape == test_df.shape

    assert not result.isna().any().any()

    for i in range(250):
        assert test_df.iloc[nan_idx_x[i], nan_idx_y[i]] == fill_values[nan_idx_y[i]]


@pytest.mark.parametrize('lower_precision',
                         [True, False])
def test_data_cleaning(lower_precision):
    raw = pd.read_table('./data/AmesHousing.txt', index_col=0)
    df = data_cleaning(raw, lower_precision)

    assert df.shape[0] > 0
    assert df.shape[1] > 0

