import numpy as np
import cvxopt


class SVMClassifier:

    def __init__(self, kernel='linear', C=None, TOL=1e-7, show_progress=False, **kwargs):
        self.fit_done = False
        self.C = C
        self.TOL = TOL
        self.show_progress = show_progress
        if kernel == 'linear':
            self.ker_mat_func = self.__ker_mat_linear
        elif kernel == 'poly':
            poly_c = kwargs.get('poly_c', 0)
            poly_d = kwargs.get('poly_d', 2)
            self.ker_mat_func = lambda x, y: self.__ker_mat_poly(x, y, poly_c, poly_d)
        elif kernel == 'rbf':
            rbf_sigma = kwargs.get('rbf_sigma', 5)
            self.ker_mat_func = lambda x, y: self.__ker_mat_rbf(x, y, rbf_sigma)

    def fit(self, X, y):
        assert not self.fit_done

        classes = np.unique(y)
        assert len(classes) == 2

        self.label_map = {-1: classes[0], 1: classes[1]}
        y = y.copy()
        y[y == classes[0]] = -1
        y[y == classes[1]] = 1

        self._lambda = self.__generate_lambda(X, y)
        self.lambda_sv = self._lambda[self._lambda > self.TOL]
        self.sv = X[self._lambda > self.TOL]
        self.svt = y[self._lambda > self.TOL]

        sv_num = len(self.sv)

        lambda_svt = self.lambda_sv * self.svt # lambda_m * t_m
        sv_ker_mat = self.ker_mat_func(self.sv, self.sv)
        self.b = (1./sv_num) * (self.svt.sum() - (sv_ker_mat @ lambda_svt).sum())

        return self

    def predict(self, X):
        lambda_svt_diag = np.diag(self.lambda_sv * self.svt)
        sv_ker_mat = self.ker_mat_func(self.sv, X)
        y_predict = np.matmul(lambda_svt_diag, sv_ker_mat)
        y_predict = y_predict.sum(axis=0)
        y_predict += self.b

        return np.array(list(map(lambda x: self.label_map[x], np.sign(y_predict))))

    def __generate_lambda(self, X, y):
        n, features = X.shape

        # http://cvxopt.org/userguide/coneprog.html#quadratic-programming
        # need to maximize L(lambda) w.r.t. lambda --> minimize -L(lambda)
        ker_mat = self.ker_mat_func(X, X)
        P = np.outer(y, y) * ker_mat

        P = cvxopt.matrix(P)
        q = cvxopt.matrix(np.ones(n) * -1)

        # sum lambda_i*y_i = 0
        A = cvxopt.matrix(y, (1,n), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None: # Non separable
            G = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else: # separable
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(n) * -1), np.identity(n))))
            h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        cvxopt.solvers.options['show_progress'] = self.show_progress
        res = cvxopt.solvers.qp(P, q, G, h, A, b)
        _lambda = np.ravel(res['x'])
        return _lambda


    def __ker_mat_linear(self, X1, X2):
        return np.matmul(X1, X2.T)

    def __ker_mat_poly(self, X1, X2, c, d):
        dp_matrix = np.matmul(X1, X2.T)
        ker_mat = np.power((dp_matrix + c), d)
        return ker_mat

    # https://stackoverflow.com/questions/37362258/creating-a-radial-basis-function-kernel-matrix-in-matlab
    # ||x1_i - x2_j||^2 = ||x_i||^2 - 2<x_i, x_j> + ||x_j||^2
    def __ker_mat_rbf(self, X1, X2, sigma):
        dp_matrix = np.matmul(X1, X2.T) * -2.
        X1_norms = np.power(X1, 2).sum(axis=1)
        X2_norms = np.power(X2, 2).sum(axis=1)
        X1_n = len(X1_norms)
        ker_mat = -(dp_matrix + X1_norms.reshape(X1_n, -1) + X2_norms)
        ker_mat = ker_mat/(2*sigma**2.)
        ker_mat = np.exp(ker_mat)
        return ker_mat