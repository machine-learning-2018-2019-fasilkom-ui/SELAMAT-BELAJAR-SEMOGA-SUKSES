from collections import defaultdict

# TODO: set to log proba
# all proba is in log scale

class MNBTextClassifier:

    def __init__(self, vocabulary=None):
        self.fit_done = False
        self.vocabulary = vocabulary
        self.prior = defaultdict(float)
        self.condprob = defaultdict(lambda: defaultdict(float))
        self.condprob_denom = defaultdict(int)  # for denominator
        self.y_count = defaultdict(int)

    # p(class=y | {term0=x[0], term1=x[1], ...} )
    # please note that you dont add new data to vocabulary when predicting
    def proba_y_given_x(self, y, x):
        assert self.fit_done

        # p(x | y)
        proba_x_given_y = defaultdict(lambda: 1.0)
        for y_val in self.y_count:
            for term in x:
                if term not in self.condprob or y_val not in self.condprob[term]:
                    proba_x_given_y[y_val] *= 1 / (self.condprob_denom[y_val] + len(self.vocabulary))
                else:
                    proba_x_given_y[y_val] *= self.condprob[term][y_val]
                assert proba_x_given_y[y_val] != 0

        result_numerator = proba_x_given_y[y] * self.proba_y(y)
        result_denumerator = sum(proba_x_given_y[y_val] * self.proba_y(y_val)
                                 for y_val in self.y_count)
        return result_numerator / result_denumerator

    # p(c)
    def proba_y(self, y):
        assert self.fit_done
        return self.prior[y]

    # update self.prior[class] as p(class=class)
    # update self.condprob[term][class] as p(term=term | class=class)
    def fit(self, X, y):
        assert(len(X) == len(y))
        assert(not self.fit_done)

        # Set up prior values and count number of labels
        for y_i in y:
            self.y_count[y_i] += 1
            self.prior[y_i] += 1

        for y_val in self.prior:
            self.prior[y_val] /= len(y)

        # Create vocabulary
        if self.vocabulary is None:
            self.vocabulary = set(word for x in X for word in x)

        # Set up intermediate value for condprob (also sets up condprob_denom)
        for x, y_i in zip(X, y):
            for term in x:
                self.condprob[term][y_i] += 1  # intermediate value (numerator) --> frequency
            self.condprob_denom[y_i] += len(x)

        # Final value for condprob
        for term in self.vocabulary:
            for y_val in self.y_count.keys():

                self.condprob[term][y_val] = (self.condprob[term][y_val] + 1) / \
                                             (self.condprob_denom[y_val] + len(self.vocabulary))

        self.fit_done = True
        # return self

    # returns: (class, prob), ... in descending order
    def predict_proba_single(self, x):
        assert self.fit_done

        class_probs = [(y_val, self.proba_y_given_x(y_val, x)) for y_val in self.y_count]
        class_probs.sort(key=lambda x: x[1], reverse=True)
        return class_probs

    def predict_proba(self, X):
        assert self.fit_done

        return [self.predict_proba_single(x) for x in X]

    def predict_single(self, x):
        assert self.fit_done

        class_probs = self.predict_proba_single(x)
        return class_probs[0][0]

    def predict(self, X):
        assert self.fit_done

        return [self.predict_single(x) for x in X]

