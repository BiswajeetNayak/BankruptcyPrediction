import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from models import *


def i_f_predict(model, X):
    pred = model.predict(X)
    return pred, model


def clf_predict(model, X):
    pred = model.predict(X)
    pred_proba = model.predict_proba(X)
    return pred, pred_proba


def save_output(pred, proba, df,model_name):
    pred_df = pd.DataFrame([pred, proba], columns=['prediction_class', 'prediction_probability'])
    pred_df = pd.concat([pred_df, df['class']], axis=1)
    pred_df.rename(columns={'class': 'actual_class'}, inplace=True)
    pred_df.to_csv(f'{config.output_file_path}predictions_{model_name}.csv')
    return None
