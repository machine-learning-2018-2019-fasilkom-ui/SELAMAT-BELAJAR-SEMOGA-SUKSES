from model.svm import SVMClassifier


def get_binary_clf(self, X, y, label_neg, label_pos, **svc_kwargs):
    X = X[(y == label_neg) | (y == label_pos)]
    y = y[(y == label_neg) | (y == label_pos)]

    clf = SVMClassifier(**svc_kwargs)
    clf.fit(X, y)
    return clf