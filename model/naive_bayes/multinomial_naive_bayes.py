from collections import defaultdict
import numpy as np

# all proba is in log scale
# binary only
class MNBTextClassifier:

    def __init__(self):
        self.fit_done = False
        self.log_prior = defaultdict(float) # log scale
        self.log_condprob = defaultdict(lambda: defaultdict(float)) # log scale
        self.condprob_denom = defaultdict(int)  # for denominator, not log scale
        self.y_count = defaultdict(int)

    # p(class=y | {term0=x[0], term1=x[1], ...} )
    # please note that you dont add new data to vocabulary when predicting
    def proba_y_given_x(self, y, x):
        assert self.fit_done

        # p(x | y), log scale
        proba_x_given_y = defaultdict(float)
        for y_val in [0, 1]:
            for term in x:
                if term not in self.log_condprob or y_val not in self.log_condprob[term]:
                    proba_x_given_y[y_val] -= np.log(self.condprob_denom[y_val] + self.vocab_len)
                else:
                    proba_x_given_y[y_val] += self.log_condprob[term][y_val]

        log_result_numerator = proba_x_given_y[y] + self.log_proba_y(y)

        # log (x+y) = log(x) + log1p(exp(log(y) - log(x)) <-- for denominator
        log_result_denom = tuple(proba_x_given_y[y_val] + self.log_proba_y(y_val) for y_val in [0, 1])
        log_result_denom_sum = log_result_denom[0] + np.log1p(np.exp(log_result_denom[1] - log_result_denom[0]))

        return log_result_numerator - log_result_denom_sum

    # p(c), log scale
    def log_proba_y(self, y):
        assert self.fit_done
        return self.log_prior[y]

    # update self.prior[class] as p(class=class)
    # update self.condprob[term][class] as p(term=term | class=class)
    def fit(self, X, y):
        assert(len(X) == len(y))
        assert(not self.fit_done)

        # Set up prior values and count number of labels
        for y_i in y:
            self.y_count[y_i] += 1
            self.log_prior[y_i] += 1

        # assert that data has only 2 classes (0 and 1)
        assert len(self.y_count) == 2
        assert 1 in self.y_count and 0 in self.y_count

        for y_val in self.log_prior:
            self.log_prior[y_val] = np.log(self.log_prior[y_val]) - np.log(len(y))

        # Set up intermediate value for condprob (also sets up condprob_denom)
        for x, y_i in zip(X, y):
            for term in x:
                self.log_condprob[term][y_i] += 1  # intermediate value (numerator) --> frequency
            self.condprob_denom[y_i] += len(x)
        self.vocab_len = len(self.log_condprob.keys()) # vocabulary

        # Final value for condprob
        for term in self.log_condprob.keys(): # vocabulary
            for y_val in [0, 1]:
                self.log_condprob[term][y_val] = np.log(self.log_condprob[term][y_val] + 1) - \
                                                 np.log(self.condprob_denom[y_val] + self.vocab_len)

        self.fit_done = True
        return self

    # returns: (class, prob), ... in descending order
    def predict_log_proba_single(self, x):
        assert self.fit_done

        class_probs = [(y_val, self.proba_y_given_x(y_val, x)) for y_val in [0, 1]]
        class_probs.sort(key=lambda x: x[1], reverse=True)
        return class_probs

    def predict_log_proba(self, X):
        assert self.fit_done

        return [self.predict_log_proba_single(x) for x in X]

    def predict_single(self, x):
        assert self.fit_done

        class_probs = self.predict_log_proba_single(x)
        return class_probs[0][0]

    def predict(self, X):
        assert self.fit_done

        return [self.predict_single(x) for x in X]

