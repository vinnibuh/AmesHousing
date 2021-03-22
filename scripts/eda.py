import pandas as pd
from sklearn.decomposition import PCA

FEATURES_TO_DROP = [
    'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF',
    '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF',
    'Garage Cars', 'Bsmt Full Bath', 'Full Bath',
    'Bsmt Half Bath', 'Half Bath', 'Bedroom AbvGr',
    'TotRms AbvGrd', 'Year Built', 'Garage Yr Blt',
    'Year Remod/Add', 'Mo Sold', 'Bsmt Qual',
    'Fireplaces', 'Garage Finish', 'Garage Qual',
    'Roof Matl', 'Heating', 'Utilities',
    'Condition 2', 'Street', 'Sale Condition',
    'Exterior 2nd'
]

ORDERED_FEATURES = [
    'Exter Qual', 'Exter Cond', 'Bsmt Exposure',
    'BsmtFin Type 1', 'BsmtFin Type 2',
    'Heating QC', 'Kitchen Qual', 'Fireplace Qu',
    'Functional', 'Central Air', 'Paved Drive'
]

CAT_FEATURES_DICT = {
    'Condition 1': {'Feedr': 'Street',
                    'Artery': 'Street',
                    'PosN': 'Pos',
                    'PosA': 'Pos',
                    'RRNn': 'RRn',
                    'RRAn': 'RRn',
                    'RRNe': 'RRe',
                    'RRAe': 'RRe'},
    'Lot Shape': {'IR1': 'IR',
                  'IR2': 'IR',
                  'IR3': 'IR'},
    'Land Slope': {'Mod': 'Yes',
                   'Sev': 'Yes',
                   'Gtl': 'No'},
    'Garage Type': {'CarPort': 'Other',
                    '2Types': 'Other',
                    'Basment': 'Other',
                    'missing': 'Other'},
    'Lot Config': {'FR2': 'FR',
                   'FR3': 'FR'},
    'House Style': {'SFoyer': 'Sep',
                    'SLvl': 'Sep'}
}

DISORDERED_FEATS = {
    'Bldg Type',
    'Condition 1',
    'Exterior 1st',
    'Foundation',
    'Garage Type',
    'House Style',
    'Land Contour',
    'Lot Config',
    'MS SubClass',
    'MS Zoning',
    'Neighborhood',
    'Roof Style'
}

PREFIXES = {
    'Bldg Type': 'bldg_type',
    'Condition 1': 'condition',
    'Exterior 1st': 'exterior',
    'Foundation': 'foundation',
    'Garage Type': 'garage_type',
    'House Style': 'house_style',
    'Land Contour': 'land_contour',
    'Lot Config': 'lot_config',
    'MS SubClass': 'ms_subclass',
    'MS Zoning': 'ms_zoning',
    'Neighborhood': 'neighbourhood',
    'Roof Style': 'roof_style'
}

PCA_FEATS = ['Overall Qual', 'Exter Qual', 'Kitchen Qual',
             'Total Bsmt SF', 'Gr Liv Area', 'Garage Area']


def convert_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_remodeled'] = (df['Year Built'] == df['Year Remod/Add']).astype('int32')
    df = df.drop(FEATURES_TO_DROP, axis=1)

    # apply ranking-based coding by mean 'SaleType' value
    ranking = dict()
    train_idx = df['SalePrice'].notna()
    df_train = df.loc[train_idx, :]

    df['bsmt_cond_dmy'] = (df['Bsmt Cond'].isin(['missing', 'Po', 'Fa'])).astype('int32')
    df['fireplace_qu_dmy'] = (df['Fireplace Qu'].isin(['Po', 'Fa'])).astype('int32')
    df['garage_cond_dmy'] = (df['Garage Cond'] == 'TA').astype('int32')
    df = df.drop(['Bsmt Cond', 'Garage Cond'], axis=1)

    for feature in ORDERED_FEATURES:
        ranking[feature] = df_train.groupby(by=feature)['SalePrice'].mean() \
            .sort_values().rank(method='first').to_dict()

    for col, value in df[ORDERED_FEATURES].iteritems():
        df[col] = value.map(ranking[col])
        df[col] = df[col].fillna(df.loc[train_idx, col].mean())
    for col in CAT_FEATURES_DICT.keys():
        df[col] = df[col].apply(lambda x: CAT_FEATURES_DICT[col].get(x, x))

    df['is_new'] = (df['Sale Type'] == 'New').astype('int32')
    df['is_SBrkr'] = (df['Electrical'] == 'SBrkr').astype('int32')
    df['has_stone_mas_vnr'] = (df['Mas Vnr Type'] == 'Stone').astype('int32')
    df['lot_is_regular'] = (df['Lot Shape'] == 'Reg').astype('int32')
    df['has_slope'] = (df['Land Slope'] == 'Yes').astype('int32')
    df = df.drop(['Fence', 'Sale Type', 'Electrical',
                  'Mas Vnr Type', 'Lot Shape', 'Land Slope'], axis=1)

    for idx, col in enumerate(DISORDERED_FEATS):
        freqs = df.loc[train_idx, col].value_counts(normalize=True)
        mapping = df[col].map(freqs)
        df[col] = df[col].mask(mapping < 0.01, 'Other')

    df = pd.get_dummies(df, columns=DISORDERED_FEATS, prefix=PREFIXES)

    pca = PCA(n_components=4)
    X_train = df.loc[train_idx, PCA_FEATS].values
    X = df[PCA_FEATS].values
    pca.fit(X_train)
    X_reduced = pca.transform(X)
    X_frame = pd.DataFrame(X_reduced, columns=['comp_1', 'comp_2', 'comp_3', 'comp_4'])
    X_frame.index = df.index
    df = pd.concat([df.iloc[:, 1:], X_frame], axis=1). \
        drop(PCA_FEATS, axis=1)
    return df
