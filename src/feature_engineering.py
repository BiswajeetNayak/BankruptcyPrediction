import pandas as pd
import numpy as np
from models import *
from predict import *
from data_preprocessing import *


def engineer_missing_val_indicators(df):
    for col in df.columns[:-1]:
        df[col + '_missing_val_ind'] = pd.Series([1 if b else 0 for b in df[col].isna()])
    return df


def outlier_indicators(df):
    outliers = []
    for year in range(1,6):
        '''df_norm = normalized_df(df.drop(['#years'],axis=1))
        df_norm = pd.concat([df['#years'],df_norm])
        print(df_norm.head())'''
        #print(year)
        X = df[df['#years'] == year].values
        #print(df['#years'].value_counts())
        #print(X.shape)
        i_f = isolation_forest(X)
        pred = i_f_predict(i_f, X)
        #print(pred.shape)
        #df.loc[df['#years'] == year, 'Outlier_Ind'] = pred[0]
        outliers.append(pred)

    df['Outlier_Ind'] = np.concatenate(outliers)

    df.loc[df['Outlier_Ind'] == 1, 'Outlier_Ind'] = 0
    df.loc[df['Outlier_Ind'] == -1, 'Outlier_Ind'] = 1
    return df, outliers
