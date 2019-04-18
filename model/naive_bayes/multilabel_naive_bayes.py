import numpy as np
import pathos.multiprocessing as mp
import sys
import time
from .multinomial_naive_bayes import MNBTextClassifier
from util.multiprocessing import sequential_execute
from util.naive_bayes import get_ovr_classifier

# Multilabel MNB Text Classifier, implemented via one vs rest
# Implemented with multiprocessing
class MultilabelMNBTextClassifier:

    def __init__(self, n_jobs=-1):
        self.fit_done = False
        self.n_jobs = mp.cpu_count()+2 if n_jobs == -1 else n_jobs
        self.classifiers = dict()

    def fit(self, X, Y):
        assert not self.fit_done
        assert len(X) == len(Y)

        possible_labels = list(set(y_val for y in Y for y_val in y))
        self.vocabulary = set(word for x in X for word in x)

        print('possible_labels:', possible_labels)
        job_labels = np.array_split(possible_labels, self.n_jobs)

        now = time.time()
        pool = mp.Pool(self.n_jobs)
        results = pool.starmap_async(sequential_execute, [(get_ovr_classifier,
                                                           [{
                                                               'X': X,
                                                               'Y': Y,
                                                               'vocabulary': self.vocabulary,
                                                               'label': lbl,
                                                               'return_label': True
                                                           } for lbl in job])
                                                          for job in job_labels]).get()
        results_flat = [label_classifier for sublist in results for label_classifier in sublist]
        self.classifiers = dict(results_flat)
        print('fit time elapsed:', (time.time() - now))
        print(self.classifiers)

        self.fit_done = True

    def predict_proba_single(self, x, max_classes=-1):
        assert self.fit_done

        class_proba = []
        for label in self.classifiers:
            proba = dict(self.classifiers[label].predict_proba_single(x))[1]
            class_proba.append((label, proba))

        class_proba.sort(key=lambda x: x[1], reverse=True)
        if max_classes == -1:
            return class_proba
        else:
            return class_proba[:max_classes]

    def predict_proba(self, X, max_classes=-1):
        assert self.fit_done
        return [self.predict_proba_single(x, max_classes=max_classes) for x in X]

