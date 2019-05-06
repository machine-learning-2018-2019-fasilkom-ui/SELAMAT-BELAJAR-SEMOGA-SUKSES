import sys
import multiprocess as mp
import numpy as np
from model.naive_bayes import MNBTextClassifier


def get_binary_clf_from_multilabel(X, Y, label, return_label=False):
    # process = mp.current_process()
    # print('nb_util: creating classifier for label', label, 'PID:', process.pid)
    # sys.stdout.flush()
    clf = MNBTextClassifier()
    y_tofit = [1 if label in y else 0
               for y in Y]
    clf.fit(X, y_tofit)

    if return_label:
        return label, clf
    return clf


def multilabel_proba_single(x, classifiers, max_classes=-1, output_id=None):
    # process = mp.current_process()
    # print('PID:', process.pid, 'id:', output_id)
    # sys.stdout.flush()
    class_proba = []
    for label in classifiers.keys():
        proba = dict(classifiers[label].predict_log_proba_single(x))[1]
        class_proba.append((label, proba))

    class_proba.sort(key=lambda x: x[1], reverse=True)
    if max_classes == -1:
        return class_proba if output_id is None else (output_id, class_proba)
    else:
        return class_proba[:max_classes] if output_id is None else class_proba


def get_predictions(label_log_probas, threshold=np.log(0.5)):
    predictions = []
    for log_probas in label_log_probas:
        to_append = []
        for label, log_proba in log_probas:
            if log_proba > threshold:
                to_append.append(label)
        predictions.append(to_append)

    return predictions