import dill


def pickle_classifier(clf, clf_filename):
    with open(clf_filename, 'wb') as pickle_file:
        dill.dump(clf, pickle_file)


def load_pickled_classifier(clf_filename):
    with open(clf_filename, 'rb') as pickle_file:
        clf = dill.load(pickle_file)

    return clf