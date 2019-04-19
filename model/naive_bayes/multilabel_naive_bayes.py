import numpy as np
from multiprocess import Manager, Queue, Process
# import multiprocess as mp
import pathos.multiprocessing as mp
import sys
import time
from .multinomial_naive_bayes import MNBTextClassifier
from util.multiprocessing import sequential_execute
from util.naive_bayes import get_binary_clf_from_multilabel

# Multilabel MNB Text Classifier, implemented via one vs rest
# Implemented with multiprocessing
# Probability with log scale
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

        # print('possible_labels:', possible_labels)
        job_labels = np.array_split(possible_labels, self.n_jobs)

        now = time.time()
        with Manager() as manager:
            X_proxy = manager.list(X)
            Y_proxy = manager.list(Y)
            vocab_proxy = manager.list(list(self.vocabulary))
            # print('getting')
            output_queue = Queue()
            processes = [Process(target=sequential_execute,
                                 args=(output_queue,
                                       get_binary_clf_from_multilabel,
                                       [{'X': X_proxy,
                                         'Y': Y_proxy,
                                         'vocabulary': vocab_proxy,
                                         'label': lbl,
                                         'return_label': True
                                         } for lbl in job]))
                         for job in job_labels]
            [p.start() for p in processes]
            results = [output_queue.get() for p in processes]
            [p.join() for p in processes]
        results_flat = [label_classifier for sublist in results for label_classifier in sublist]
        self.classifiers = dict(results_flat)
        print('fit time elapsed:', (time.time() - now))
        print(self.classifiers)

        self.fit_done = True

    def predict_log_proba_single(self, x, max_classes=-1):
        assert self.fit_done

        class_proba = []
        for label in self.classifiers:
            proba = dict(self.classifiers[label].predict_log_proba_single(x))[1]
            class_proba.append((label, proba))

        class_proba.sort(key=lambda x: x[1], reverse=True)
        if max_classes == -1:
            return class_proba
        else:
            return class_proba[:max_classes]

    def predict_log_proba(self, X, max_classes=-1):
        assert self.fit_done
        return [self.predict_log_proba_single(x, max_classes=max_classes) for x in X]

