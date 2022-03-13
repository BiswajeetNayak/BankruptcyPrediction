import pandas as pd
import numpy as np
from models import *
from predict import *


def engineer_missing_val_indicators(df):
    for col in df.columns:
        df[col + '_missing_val_ind'] = pd.Series([1 if b else 0 for b in df[col].isna()])
    return df


def outlier_indicators(df):
    outliers = []
    for year in range(1, 6):
        X = df[df['#years'] == year].iloc[:, :-1].values
        i_f = isolation_forest(X)
        pred = i_f_predict(i_f, X)
        df.loc[df['#years'] == year, 'Outlier_Ind'] = pred[0]
        outliers.append(pred)
    df.loc[df['Outlier_Ind'] == 1, 'Outlier_Ind'] = 0
    df.loc[df['Outlier_Ind'] == -1, 'Outlier_Ind'] = 1
    return df, outliers
