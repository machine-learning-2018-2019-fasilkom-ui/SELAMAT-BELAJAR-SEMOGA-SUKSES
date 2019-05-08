from sklearn.metrics import classification_report
import numpy as np


def get_clf_reports_per_label(multilabel_clf, X_test, Y_test):

    clf_reports = dict()
    for label, clf in multilabel_clf.classifiers.items():
        y_test = np.array([1 if label in y else 0 for y in Y_test])
        y_pred = clf.predict(X_test)
        clf_reports[label] = classification_report(y_test, y_pred)

    return clf_reports