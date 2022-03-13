filepath_1year = r'../data/input/1year.arff'
filepath_2year = r'../data/input/2year.arff'
filepath_3year = r'../data/input/3year.arff'
filepath_4year = r'../data/input/4year.arff'
filepath_5year = r'../data/input/5year.arff'

attribute_file_path = r'../data/input/attribute_information.txt'
models_file_path = r'../models'
output_file_path = r'../output'

isolation_forest_contamination = 0.01

test_split_perc = 0.2

models = ['LogisticRegression', 'RandomForest', 'AdaBoost', 'GBM', 'SVM']
CV_FOLD = 5

eval_metrics = ['confusion_matrix', 'classification_report', 'roc_auc_score', 'recall_score', 'f1_score']
