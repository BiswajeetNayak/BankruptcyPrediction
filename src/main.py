import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.model_selection import train_test_split, cross_val_score
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

    ## Creating dummy vars for missing value indicators.
    data_dummy = engineer_missing_val_indicators(data.drop(['class'], axis=1))
    data_dummy = impute_missing_vals(data_dummy)
    data_dummy['#years'] = data_dummy['#years'].astype('int')
    data_dummy, outlier = outlier_indicators(data_dummy)

    data_dummy = pd.concat([data_dummy, data['class']], axis=1)

    models, predictions, probabilities, evaluation_results, mean_cv_score = [], [], [], [], []
    model_all_data, pred_all_data, proba_all_data, eval_results_all_data, cv_score_all_data = model_training(data_dummy)

    for year in range(1, 6):
        model, pred, proba, eval_results, cv_score = model_training(data_dummy[data_dummy['#years'] == year])

        #save_output(pred,proba,data_dummy[data_dummy['#years'] == year])

print('Execution Completed. \n All models are trained and predictions and probabilities are saved in data/output folder.')
