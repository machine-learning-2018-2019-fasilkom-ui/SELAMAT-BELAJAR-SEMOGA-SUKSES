from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def print_clf_reports_per_label(multilabel_clf, X_test, Y_test):

    for label, clf in multilabel_clf.classifiers.items():
        print('-----------------------', label)
        y_test = np.array([1 if label in y else 0 for y in Y_test])
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)
        print('ACCURACY:', accuracy_score(y_test, y_pred))
