from model.svm import SVMClassifier
import numpy as np

def test_svm():
    X = np.zeros((100,2))
    np.random.seed(64)
    X[0:50,0] = np.random.uniform(-3,1,50)
    np.random.seed(36)
    X[0:50,1] = np.random.uniform(-3,1,50)

    np.random.seed(64)
    X[50:,0] = np.random.uniform(2,4,50)
    np.random.seed(36)
    X[50:,1] = np.random.uniform(-2,6,50)
    y = np.array([-1 if i < 50 else 1 for i in range(100)])

    svm = SVMClassifier(kernel='poly', poly_c=1, poly_d=3)
    svm.fit(X,y)
    return X,y,svm

if __name__ == '__main__':
    X, y, svm = test_svm()

    g = np.array([[0.95732021, -1.0710335],
                  [0.82783717, 0.2080372],
                  [2.00389699, -0.31674417],
                  [2.07133994, 3.36711104],
                  [2.03597597, 0.73548602]])
    t = np.array([-1, -1, 1, 1, 1])

    X_test = np.array([[0.53223423, 4.2422343242]])
    y_test, _ = svm.predict(X_test)

    np.testing.assert_array_almost_equal(svm.sv, g)
    np.testing.assert_array_almost_equal(svm.svt, t)
    np.testing.assert_array_almost_equal(y_test, np.array([-1.]))

    print('Testing OK')
