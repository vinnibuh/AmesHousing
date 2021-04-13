"""Classes and functions used for EDA step of research"""
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

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
    'Exterior 2nd', 'Fence', 'Sale Type',
    'Electrical', 'Mas Vnr Type', 'Lot Shape',
    'Land Slope', 'Bsmt Cond', 'Garage Cond'
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

DISORDERED_FEATS = [
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
]

PREFIXES = {
    'Bldg Type': 'dmy_bldg_type',
    'Condition 1': 'dmy_condition',
    'Exterior 1st': 'dmy_exterior',
    'Foundation': 'dmy_foundation',
    'Garage Type': 'dmy_garage_type',
    'House Style': 'dmy_house_style',
    'Land Contour': 'dmy_land_contour',
    'Lot Config': 'dmy_lot_config',
    'MS SubClass': 'dmy_ms_subclass',
    'MS Zoning': 'dmy_ms_zoning',
    'Neighborhood': 'dmy_neighbourhood',
    'Roof Style': 'dmy_roof_style'
}

PCA_FEATS = ['Overall Qual', 'Exter Qual', 'Kitchen Qual',
             'Total Bsmt SF', 'Gr Liv Area', 'Garage Area']


class HousingTransformer(BaseEstimator, TransformerMixin):
    """A custom Transformer to process data. Includes ranking categorical features, making dummies and PCA."""

    def __init__(self, n_pca_components=4, pca_cols=None):
        """
        Initialise HousingTransformer. Takes as parameters number of PCA components to leave and subset
        of columns on which PCA will be fit.

        :param n_pca_components: number of PCA components
        :param pca_cols: subset of columns, list
        """
        if pca_cols is None:
            self.pca_cols = PCA_FEATS
        else:
            self.pca_cols = pca_cols
        self.ranking = dict()
        self.freqs = dict()
        self.means = dict()
        self.pca = PCA(n_components=n_pca_components)
        self.base_dummy_cols = []
        self.n_pca_components = n_pca_components

    def fit(self, df: pd.DataFrame, y: pd.DataFrame):
        df = pd.concat([df, y], axis=1)
        df.columns = [*df.columns[:-1], 'SalePrice']
        for col, value in df[ORDERED_FEATURES].iteritems():
            self.ranking[col] = df.groupby(by=col)['SalePrice'].mean() \
                .sort_values().rank(method='first').to_dict()
            df[col] = value.map(self.ranking[col])

        for idx, col in enumerate(DISORDERED_FEATS):
            freqs = df[col].value_counts(normalize=True).to_dict()
            mapping = df[col].map(freqs)
            df[col] = df[col].mask(mapping < 0.01, 'Other')

            self.freqs[col] = defaultdict(default_freq, freqs)

        df = pd.get_dummies(df, columns=DISORDERED_FEATS, prefix=PREFIXES)
        self.base_dummy_cols = df.filter(regex='dmy_+', axis=1).columns
        self.means = df[ORDERED_FEATURES].mean(axis=0).to_dict()

        self.pca.fit(df[self.pca_cols])
        return self

    def transform(self, df: pd.DataFrame, y=None):
        df = df.copy()
        for col, value in df[ORDERED_FEATURES].iteritems():
            df.loc[:, col] = value.map(self.ranking[col])
            df.loc[:, col] = df[col].fillna(self.means[col])

        for idx, col in enumerate(DISORDERED_FEATS):
            mapping = df[col].map(self.freqs[col])
            df.loc[:, col] = df[col].mask(mapping < 0.01, 'Other')

        df = self.add_dummies(df)
        df = transform_pca(df, self.pca, self.pca_cols)
        return df.values

    def add_dummies(self, df):
        """
        Create dummies in dataframe based on already determined columns. Pays attention to possibility of
        difference in unique values between data which was used to fit transformer and data to be transformed.

        :param df: source dataframe
        :return: dataframe, same length
        """

        df_dummies = pd.get_dummies(df.loc[:, DISORDERED_FEATS], prefix=PREFIXES)
        dummy_cols = df_dummies.columns
        df_len = df_dummies.shape[0]

        dummies_to_drop = set(dummy_cols).difference(set(self.base_dummy_cols))
        dummies_to_add = set(self.base_dummy_cols).difference(set(dummy_cols))

        df_dummies = df_dummies.drop(list(dummies_to_drop), axis=1)

        dummies_to_add_size = (df_len, len(dummies_to_add))
        dummies_to_add_data = np.zeros(dummies_to_add_size)
        df_dummies_to_add = pd.DataFrame(data=dummies_to_add_data,
                                         columns=list(dummies_to_add))
        df_dummies = pd.concat([df_dummies, df_dummies_to_add], axis=1)
        df_dummies = df_dummies.loc[:, self.base_dummy_cols]

        df = pd.concat([df.drop(DISORDERED_FEATS, axis=1), df_dummies], axis=1)
        return df


def make_binary_features(df):
    """
    Create binary features in dataframe for predefined columns. New columns are appended
    to other columns.

    :param df: source dataframe
    :return: dataframe, same length
    """
    df = df.copy()

    necessary_cols = ['Year Built', 'Bsmt Cond', 'Fireplace Qu',
                      'Garage Cond', 'Sale Type', 'Electrical',
                      'Mas Vnr Type', 'Lot Shape', 'Land Slope']

    assert set(necessary_cols).issubset(set(df.columns))

    df['is_remodeled'] = df['Year Built'].eq(df['Year Remod/Add']).astype('int32')
    df['bsmt_cond_dmy'] = (df['Bsmt Cond'].isin(['missing', 'Po', 'Fa'])).astype('int32')
    df['fireplace_qu_dmy'] = (df['Fireplace Qu'].isin(['Po', 'Fa'])).astype('int32')
    df['garage_cond_dmy'] = (df['Garage Cond'] == 'TA').astype('int32')

    df['is_new'] = (df['Sale Type'] == 'New').astype('int32')
    df['is_SBrkr'] = (df['Electrical'] == 'SBrkr').astype('int32')
    df['has_stone_mas_vnr'] = (df['Mas Vnr Type'] == 'Stone').astype('int32')
    df['lot_is_regular'] = (df['Lot Shape'] == 'Reg').astype('int32')
    df['has_slope'] = (df['Land Slope'] == 'Yes').astype('int32')

    return df


def transform_pca(df, pca, cols):
    """
    Apply PCA on subset of columns in dataframe. PCA should be already fitted.

    :param df: source dataframe
    :param pca: instance of PCA from sklearn, already fitted
    :param cols: subset of columns in form of list
    :return: dataframe, same length
    """
    X = df.loc[:, cols].values
    X_reduced = pca.transform(X)
    X_frame = pd.DataFrame(X_reduced, columns=['comp_' + str(idx) for idx in range(pca.n_components)])
    X_frame.index = df.index
    df = pd.concat([df, X_frame], axis=1). \
        drop(cols, axis=1)
    return df


def basic_feature_engineering(df):
    """
    Conduct basic feature engineering, which can be made on whole dataset without fear of data leak.
    Involves creating binary features, dropping columns and mapping several categories' values.

    :param df: dataframe to preprocess
    :return: dataframe, same length
    """
    df = make_binary_features(df)
    for col in CAT_FEATURES_DICT.keys():
        df[col] = df[col].apply(lambda x: CAT_FEATURES_DICT[col].get(x, x))
    df = df.drop(FEATURES_TO_DROP, axis=1)

    return df

def default_freq():
    return 0