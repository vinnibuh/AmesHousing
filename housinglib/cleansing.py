COLS_TO_FILTER = [
    'SalePrice', 'Garage Area', 'Total Bsmt SF',
    'BsmtFin SF 1', 'BsmtFin SF 2', 'Misc Val',
    'Pool Area', 'Screen Porch', '3Ssn Porch',
    'Enclosed Porch', 'Open Porch SF', 'Wood Deck SF',
    'Lot Area', 'Lot Frontage'
]


def data_cleaning(df, drop_pid=True, lower_precision=True, drop_rare_values=True, csv_out=None):
    initial_data_size = df.shape[0]
    df = df.drop(2261)

    if drop_pid:
        df = df.drop('PID', axis=1)

    df['MS SubClass'] = df['MS SubClass'].astype('object')

    # filter outliers
    mask = df.notna().any(axis=1)
    for col in COLS_TO_FILTER:
        threshold = df[col].quantile(0.997)
        mask = mask & ~(df[col] > threshold)
    df = df.loc[mask, :]

    if drop_rare_values:
        df.drop(['Pool QC', 'Misc Feature', 'Alley',
                 '3Ssn Porch', 'Pool Area'],
                axis=1, inplace=True)

    if lower_precision:
        df = drop_precision(df)

    # separately process columns of different types, except of 'SaleType'
    cat_feats_df = df.select_dtypes(include='object')
    df.loc[:, cat_feats_df.columns] = cat_feats_df.fillna('missing').astype('string')

    real_feats_df = df.select_dtypes(exclude='object')
    df.loc[:, real_feats_df.columns] = fill_na_real(real_feats_df)
    df = df.reset_index(drop=True)
    cleaned_data_size = df.shape[0]
    print('Initial data size: ', initial_data_size)
    print('Dropped {} lines'.format(initial_data_size-cleaned_data_size))
    print('Final data size: ', cleaned_data_size)

    if csv_out is not None:
        df.to_csv(csv_out)

    return df


def fill_na_real(real_df):
    val_counts_float_cols = real_df.nunique()

    few_unique_vals = val_counts_float_cols[val_counts_float_cols <= 10].index
    many_unique_vals = val_counts_float_cols[val_counts_float_cols > 10].index

    real_df[few_unique_vals] = real_df[few_unique_vals].fillna(real_df.median())
    real_df[many_unique_vals] = real_df[many_unique_vals].fillna(real_df.mean())
    return real_df


def drop_precision(df, new_float_type='float32', new_int_type='int32'):
    df = df.copy()

    float_cols = df.select_dtypes(include='float64').columns
    int_cols = df.select_dtypes(include='int64').columns

    df.loc[:, float_cols] = df.loc[:, float_cols].astype(new_float_type)
    df.loc[:, int_cols] = df.loc[:, int_cols].astype(new_int_type)

    return df
