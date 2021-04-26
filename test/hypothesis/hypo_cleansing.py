from hypothesis import given, assume, strategies as st
from hypothesis.extra.pandas import column, data_frames

import pandas as pd
import numpy as np

from housinglib.cleansing import fill_na_real, drop_precision


@given(data_frames(columns=[
    column(name='first',
           elements=st.floats(allow_nan=True)),
    column(name='second',
           elements=st.floats(allow_nan=True))
])
)
def test_fill_na_real_hypo(frame):
    result = fill_na_real(frame)
    assert isinstance(result, pd.DataFrame)
    assert result.notna().all(axis=None)


@given(data_frames(columns=[
    column(name='first',
           dtype='int64'),
    column(name='second',
           dtype='float64')
])
)
def test_drop_precision_hypo(frame):
    result = drop_precision(frame, 'float64', 'int64')
    assert np.allclose(result.values, frame.values, equal_nan=True)
