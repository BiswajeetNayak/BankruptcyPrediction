from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import config
from predict import *
from eval import *


def isolation_forest(X):
    # contamination = config.isolation_forest_contamination
    i_f = IsolationForest(random_state=100)
    i_f.fit(X)
    return i_f


def logistic_regression(X, y):
    lm = LogisticRegression(random_state=100, class_weight='balanced')
    lm.fit(X, y)
    return lm


def random_forest(X, y):
    rf = RandomForestClassifier(random_state=100, class_weight='balanced')
    rf.fit(X, y)
    return rf


def adaboost_clf(X, y):
    ab_clf = AdaBoostClassifier(random_state=100)
    ab_clf.fit(X, y)
    return ab_clf


def GBM(X, y):
    gbm = GradientBoostingClassifier(random_state=100)
    gbm.fit(X, y)
    return gbm


def SVM(X, y):
    svm = SVC(random_state=100, class_weight='balanced', probability=True)
    svm.fit(X, y)
    return svm


def save_model(model, model_name, path=config.models_file_path):
    joblib.dump(model, f'{model_name}_BaseModel.pkl')
    return None


def model_training(df):
    # The dataframe passed as argument is already filtered for year.
    # Each year will have a different model trained.

    X = df.drop(['class'], axis=1)
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=config.test_split_perc,
                                                        random_state=100)

    print('Distribution of classes in test data: ', np.unique(y_test, return_counts=True))

    models, predictions, probabilities, evaluation_results, mean_cv_score = [], [], [], [], []

    for model in config.models:
        if model == 'LogisticRegression':
            lm = logistic_regression(X_train, y_train)
            pred, proba = clf_predict(lm, X_test)
            cv_score = cross_validation_score(model=lm, X=X, y=y)
            save_model(lm, model)
            # save_output(pred, proba, pd.DataFrame(X_test, columns=df.columns.drop('class')),model)
            print(f'Mean {config.CV_FOLD}-Fold cross validation score for a '
                  f'{model} model is : {cv_score}')
            eval_results = []
            for metric in config.eval_metrics:
                print(f'The {metric} for {model} is: {eval_predictions(y_test, pred, metric)}')
                eval_results.append(eval_predictions(y_test, pred, metric))
            models.append(lm)
            predictions.append(pred)
            probabilities.append(proba)
            evaluation_results.append(eval_results)
            mean_cv_score.append(cv_score)
            print("============================================================================================")

        if model == 'RandomForest':
            rf = random_forest(X_train, y_train)
            pred, proba = clf_predict(lm, X_test)
            cv_score = cross_validation_score(model=lm, X=X, y=y)
            save_model(lm, model)
            # save_output(pred, proba, pd.DataFrame(X_test, columns=df.columns.drop('class')),model)
            print(f'Mean {config.CV_FOLD}-Fold cross validation score for a '
                  f'{model} model is : {cv_score}')
            eval_results = []
            for metric in config.eval_metrics:
                print(f'The {metric} for {model} is: {eval_predictions(y_test, pred, metric)}')
                eval_results.append(eval_predictions(y_test, pred, metric))
            models.append(rf)
            predictions.append(pred)
            probabilities.append(proba)
            evaluation_results.append(eval_results)
            mean_cv_score.append(cv_score)

        print("============================================================================================")

        if model == 'AdaBoost':
            lm = adaboost_clf(X_train, y_train)
            pred, proba = clf_predict(lm, X_test)
            cv_score = cross_validation_score(model=lm, X=X, y=y)
            save_model(lm, model)
            # save_output(pred, proba, pd.DataFrame(X_test, columns=df.columns.drop('class')),model)
            print(f'Mean {config.CV_FOLD}-Fold cross validation score for a '
                  f'{model} model is : {cv_score}')
            eval_results = []
            for metric in config.eval_metrics:
                print(f'The {metric} for {model} is: {eval_predictions(y_test, pred, metric)}')
                eval_results.append(eval_predictions(y_test, pred, metric))
            models.append(model)
            predictions.append(pred)
            probabilities.append(proba)
            evaluation_results.append(eval_results)
            mean_cv_score.append(cv_score)

        print("============================================================================================")

        if model == 'GBM':
            lm = GBM(X_train, y_train)
            pred, proba = clf_predict(lm, X_test)
            cv_score = cross_validation_score(model=lm, X=X, y=y)
            save_model(lm, model)
            # save_output(pred, proba, pd.DataFrame(X_test, columns=df.columns.drop('class')),model)
            print(f'Mean {config.CV_FOLD}-Fold cross validation score for a '
                  f'{model} model is : {cv_score}')
            eval_results = []
            for metric in config.eval_metrics:
                print(f'The {metric} for {model} is: {eval_predictions(y_test, pred, metric)}')
                eval_results.append(eval_predictions(y_test, pred, metric))
            models.append(model)
            predictions.append(pred)
            probabilities.append(proba)
            evaluation_results.append(eval_results)
            mean_cv_score.append(cv_score)

        print("============================================================================================")

        if model == 'SVM':
            lm = SVM(X_train, y_train)
            pred, proba = clf_predict(lm, X_test)
            cv_score = cross_validation_score(model=lm, X=X, y=y)
            save_model(lm, model)
            # save_output(pred, proba, pd.DataFrame(X_test, columns=df.columns.drop('class')),model)
            print(f'Mean {config.CV_FOLD}-Fold cross validation score for a '
                  f'{model} model is : {cv_score}')
            eval_results = []
            for metric in config.eval_metrics:
                print(f'The {metric} for {model} is: {eval_predictions(y_test, pred, metric)}')
                eval_results.append(eval_predictions(y_test, pred, metric))
            models.append(model)
            predictions.append(pred)
            probabilities.append(proba)
            evaluation_results.append(eval_results)
            mean_cv_score.append(cv_score)

        print("============================================================================================")
        return models, predictions, probabilities, evaluation_results, mean_cv_score
