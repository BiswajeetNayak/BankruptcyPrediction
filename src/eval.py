import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,recall_score,f1_score
import numpy as np
from sklearn.model_selection import cross_val_score
import config


def cross_validation_score(model, X, y, CV=config.CV_FOLD):
    cv_score = cross_val_score(estimator=model, X=X, y=y, cv=CV)
    return np.mean(cv_score)

def eval_predictions(y_true,y_pred,metric):
    if metric == 'confusion_matrix':
        return confusion_matrix(y_true,y_pred)
    if metric == 'classification_report':
        return classification_report(y_true, y_pred)
    if metric == 'roc_auc_score':
        return roc_auc_score(y_true,y_pred)
    if metric == 'recall_score':
        return recall_score(y_true,y_pred)
    if metric == 'f1_score':
        return f1_score(y_true,y_pred)

    else:
        print(f'The metric passed in the argument is not supported. '
              f'\n The following metrics in the list are supported: {config.eval_metrics}')

