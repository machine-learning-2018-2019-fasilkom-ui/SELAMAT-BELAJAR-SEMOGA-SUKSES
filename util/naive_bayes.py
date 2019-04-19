import sys
import multiprocess as mp
from model.naive_bayes import MNBTextClassifier


def get_binary_clf_from_multilabel(X, Y, vocabulary, label, return_label=False):
    process = mp.current_process()
    print('nb_util: creating classifier for label', label, 'PID:', process.pid)
    sys.stdout.flush()
    clf = MNBTextClassifier(vocabulary=vocabulary)
    y_tofit = [1 if label in y else 0
               for y in Y]
    clf.fit(X, y_tofit)

    if return_label:
        return label, clf
    return clf


def multilabel_proba(x, classifiers, max_classes=-1):
    class_proba = []
    for label in classifiers:
        proba = dict(classifiers[label].predict_log_proba_single(x))[1]
        class_proba.append((label, proba))

    class_proba.sort(key=lambda x: x[1], reverse=True)
    if max_classes == -1:
        return class_proba
    else:
        return class_proba[:max_classes]