import numpy as np
from model.naive_bayes import MNBTextClassifier

if __name__ == '__main__':
    X = np.array([
        ['chinese', 'beijing', 'chinese'],
        ['chinese', 'chinese', 'shanghai'],
        ['chinese', 'macao'],
        ['tokyo', 'japan', 'chinese']
    ])

    # class 1 is china
    # class 0 is japan
    y = np.array([1, 1, 1, 0])

    clf = MNBTextClassifier()
    clf.fit(X, y)

    np.testing.assert_almost_equal(clf.log_condprob['chinese'][1], np.log(3 / 7))
    np.testing.assert_almost_equal(clf.log_condprob['japan'][1], np.log(1 / 14))
    np.testing.assert_almost_equal(clf.log_condprob['chinese'][0], np.log(2 / 9))
    np.testing.assert_almost_equal(clf.log_condprob['japan'][0], np.log(2 / 9))

    np.testing.assert_almost_equal(clf.proba_y_given_x(1, ['asdf']), np.log(0.6585365853658537))

    np.testing.assert_almost_equal(clf.proba_y_given_x(1, ['chinese', 'chinese', 'chinese', 'tokyo', 'japan']),
                                   np.log(0.6897586117634673))
    np.testing.assert_almost_equal(
        clf.proba_y_given_x(1, ['asdfasdfasdf', 'asdfasdfasdf', 'chinese', 'chinese', 'chinese', 'tokyo', 'japan']),
        np.log(0.47884402067520343))

    np.testing.assert_almost_equal(clf.proba_y_given_x(0, ['chinese', 'chinese', 'chinese', 'tokyo', 'japan']),
                                   np.log(0.31024138823653263))

    np.testing.assert_almost_equal(clf.log_proba_y(1), np.log(3 / 4))
    np.testing.assert_almost_equal(clf.log_proba_y(0), np.log(1 / 4))
    np.testing.assert_almost_equal(clf.predict_single(['chinese', 'chinese', 'tokyo', 'japan']), 1)

    print('Testing OK')