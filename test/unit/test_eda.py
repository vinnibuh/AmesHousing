import pytest
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sys import path

path.append('.')

import housinglib.eda as eda
import housinglib.cleansing as cleansing


@pytest.mark.parametrize('n_components',
                         [1, 2, 3, 4, 5, 6])
def test_basic_transform_pca(n_components):
    rng = np.random.default_rng()
    col_names = ['f_' + str(n) for n in range(20)]
    source_df = pd.DataFrame(rng.random((100, 20), dtype='float64'),
                             columns=col_names)
    test_cols = ['f_' + str(x) for x in rng.choice(20, 10, replace=False)]
    pca = PCA(n_components=n_components)
    X = source_df.loc[:, test_cols]
    pca.fit(X.values)
    test_df = eda.transform_pca(source_df, pca, test_cols)

    columns_diff = len(test_cols) - n_components
    assert source_df.shape[1] - test_df.shape[1] == columns_diff
    assert source_df.shape[0] == test_df.shape[0]

    transformed_col_names = ['comp_' + str(x) for x in range(n_components)]
    assert (test_df.columns[-n_components:] == transformed_col_names).all()


def test_make_binary_features():
    raw = pd.read_table('./data/AmesHousing.txt', index_col=0)
    source_df = cleansing.data_cleaning(raw)

    test_df = eda.make_binary_features(source_df)

    assert test_df.shape[0] == source_df.shape[0]
    assert test_df.shape[1] - source_df.shape[1] == 9
    assert test_df.iloc[:, -9:].notna().all(axis=None)
    assert test_df.iloc[:, -9:].isin([0, 1]).all(axis=None)


def test_basic_feature_engineering():
    raw = pd.read_table('./data/AmesHousing.txt', index_col=0)
    source_df = cleansing.data_cleaning(raw)

    test_df = eda.basic_feature_engineering(source_df)

    assert test_df.shape[0] == source_df.shape[0]
    assert set(eda.FEATURES_TO_DROP).isdisjoint(set(test_df.columns))

    def if_values_mapped(x):
        key = x.name
        new_unique_values = set(x.values)
        old_unique_values = set(eda.CAT_FEATURES_DICT[key].keys())
        return new_unique_values.isdisjoint(old_unique_values)

    cat_feats = set(eda.CAT_FEATURES_DICT.keys())
    not_dropped_cat_feats = cat_feats.difference(set(eda.FEATURES_TO_DROP))
    mapped_columns_df = test_df.loc[:, list(not_dropped_cat_feats)]
    assert mapped_columns_df.apply(if_values_mapped, axis=0, raw=False).all()
