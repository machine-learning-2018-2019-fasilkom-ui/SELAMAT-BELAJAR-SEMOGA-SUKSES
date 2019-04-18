import sys
from model.naive_bayes import MNBTextClassifier


def get_ovr_classifier(X, Y, vocabulary, label, return_label=False):
    print('nb_util: creating classifier for label', label)
    sys.stdout.flush()
    clf = MNBTextClassifier(vocabulary=vocabulary)
    y_tofit = [1 if label in y else 0
               for y in Y]
    clf.fit(X, y_tofit)

    if return_label:
        return label, clf
    return clf
