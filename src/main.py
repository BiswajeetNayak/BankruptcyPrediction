import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import warnings

from data_preprocessing import *
from feature_engineering import *
from models import *
from eval import *
from predict import *
import config

warnings.filterwarnings('ignore')
plt.style.use('ggplot')


def read_file(path):
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    df.iloc[:, -1] = df.iloc[:, -1].astype('int32')
    return df


def get_attribute_from_file(path, sep='\t'):
    data = pd.read_csv(path, sep=sep, header=None)
    attributes = data.iloc[:, 1].tolist()
    return attributes


if __name__ == '__main__':
    data_1year = read_file(config.filepath_1year)
    data_2year = read_file(config.filepath_2year)
    data_3year = read_file(config.filepath_3year)
    data_4year = read_file(config.filepath_4year)
    data_5year = read_file(config.filepath_5year)

    columns = get_attribute_from_file(config.attribute_file_path)
    # print('Total number of columns in the file is: ', len(columns))

    list_dfs = [data_5year, data_4year, data_2year, data_3year, data_1year]

    data = concat_dfs(list_dfs)
    data = rename_columns(data, columns)

    missing_value_df = pd.DataFrame({'column_name': data.columns,
                                     'percent_missing': 100 * data.isnull().sum() / data.shape[0]})

    missing_value_df = missing_value_df.sort_values('percent_missing', ascending=False)
