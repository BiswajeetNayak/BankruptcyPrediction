import pandas as pd


def rename_columns(df, cols):
    df.columns.values[:-2] = cols
    return df


def concat_dfs(df_list):
    for idx, df in enumerate(df_list):
        df['#years'] = len(df_list) - idx

    df = pd.concat(df_list, axis=0)
    return df
