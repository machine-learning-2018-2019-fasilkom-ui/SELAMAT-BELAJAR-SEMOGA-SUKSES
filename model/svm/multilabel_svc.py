import numpy as np
from util.svm import get_binary_clf
from .svc import SVMClassifier
from collections import defaultdict
import pathos.multiprocessing as mp
from multiprocess import Manager, Queue, Process
import itertools
from scipy.stats import mode
from .svc_large import LargeSVMClassifier
import time

# Multilabel MNB Text Classifier, implemented via one vs one
# Also implemented with multiprocessing TODO
class MultilabelSVMClassifier:

    def __init__(self, n_jobs=-1, **kwargs):
        self.svc_kwargs = kwargs
        self.fit_done = False
        #self.n_jobs = mp.cpu_count()+2 if n_jobs == -1 else n_jobs TODO

    def fit(self, X, Y):
        assert not self.fit_done

        self.labels = np.unique([label for y in Y for label in y])
        assert len(self.labels) > 1

        self.classifiers = dict()
        for label in self.labels:
            now = time.time()
            print('multilabel: fitting', label)
            y_tofit = np.array([1 if label in y else 0 for y in Y])
            svc = LargeSVMClassifier(3, 3000, **self.svc_kwargs)
            svc.fit(X, y_tofit)
            self.classifiers[label] = svc
            print('label fit', label, 'done in', (time.time()-now), 'seconds')

        self.fit_done = True

    def predict(self, X):
        assert self.fit_done

        pred_labels = []
        for i in range(len(X)):
            pred_labels.append([])

        for label, clf in self.classifiers.items():
            preds = clf.predict(X)
            for idx, pred in enumerate(preds):
                if pred == 1:
                    pred_labels[idx].append(label)

        return pred_labels

    # Get one vs one classifier


