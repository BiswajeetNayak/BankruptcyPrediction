import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def rename_columns(df, cols):
    df.columns.values[:-2] = cols
    return df


def concat_dfs(df_list):
    for idx, df in enumerate(df_list):
        df['#years'] = len(df_list) - idx

    df = pd.concat(df_list, axis=0)
    return df


def impute_missing_vals(df):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imp.fit_transform(df.values)
    df = pd.DataFrame(X, index=df.index, columns=df.columns)
    return df


