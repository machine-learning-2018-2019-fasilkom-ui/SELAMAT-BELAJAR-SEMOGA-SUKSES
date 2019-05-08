from .svc import SVMClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import mode

# SVMClassifier for large data, uses random sampling on training set and
class LargeSVMClassifier:

    def __init__(self, num_svms, max_data_per_train=1000, **kwargs):
        assert num_svms % 2 == 1

        self.num_svms = num_svms
        self.svc_kwargs = kwargs
        self.svcs = []
        self.max_data_per_train = max_data_per_train
        self.fit_done = False

    def fit(self, X, y):
        assert not self.fit_done
        n = len(X)
        train_ratio = min(self.max_data_per_train,n)/n
        for i in range(self.num_svms):
            X_train, _, y_train, _ = train_test_split(X, y, train_size=train_ratio, test_size=1-train_ratio, stratify=y)
            svc = SVMClassifier(**self.svc_kwargs)
            svc.fit(X_train, y_train)
            self.svcs.append(svc)

        self.fit_done = True

    def predict(self, X):
        assert self.fit_done
        preds = np.array([svc.predict(X) for svc in self.svcs])
        return mode(preds)[0][0]
