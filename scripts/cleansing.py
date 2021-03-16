from collections import defaultdict

def data_cleaning(df, drop_pid=True, lower_precision=True, drop_rare_values=True):
    
    if drop_pid:
        df = df.drop('PID', axis=1)
        
    # filter outliers
    cols_to_filter = ['SalePrice', 'Garage Area', 'Total Bsmt SF', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Misc Val']
    for col in cols_to_filter:
        threshold = df[col].quantile(0.997)
        df = df.loc[df[col] < threshold, :]
    
    if drop_rare_values:
        df.drop(['Pool QC', 'Misc Feature', 'Alley'], axis=1, inplace=True)
    
    float_type = 'float64'
    int_type = 'int64'
    if lower_precision:
        float_cols = df.select_dtypes(include=float_type).columns
        int_cols = df.select_dtypes(include=int_type).columns
        
        new_float_type, new_int_type = 'float32', 'int32'
        df.loc[:, float_cols] = df.loc[:, float_cols].astype(new_float_type)
        df.loc[:, int_cols] = df.loc[:, int_cols].astype(new_int_type)
        float_type, int_type = new_float_type, new_int_type
    
    # separately process columns of different types
    col_by_type = defaultdict(list)
    for index, value in df.dtypes.items():
        col_by_type[str(value)].append(index)
        
    df.loc[:, col_by_type['object']] = df.loc[:, col_by_type['object']].fillna('missing')
    
    # floats
    float_cols = col_by_type[float_type]
    val_counts_float_cols = df[float_cols].nunique()
    
    few_unique_vals = val_counts_float_cols[val_counts_float_cols<=10].index
    many_unique_vals = val_counts_float_cols[val_counts_float_cols>10].index
    
    df[few_unique_vals] = df[few_unique_vals].fillna(df.median()).astype(int_type)
    df[many_unique_vals] = df[many_unique_vals].fillna(df.mean())
    
    # ints
    int_cols = col_by_type[int_type]
    val_counts_int_cols = df[int_cols].nunique()
    
    few_unique_vals = val_counts_int_cols[val_counts_int_cols<=10].index
    many_unique_vals = val_counts_int_cols[val_counts_int_cols>10].index

    df[few_unique_vals] = df[few_unique_vals].fillna(df.median())
    df[many_unique_vals] = df[many_unique_vals].fillna(df.mean())
    
    return df