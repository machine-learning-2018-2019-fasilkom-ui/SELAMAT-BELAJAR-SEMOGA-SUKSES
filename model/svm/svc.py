import numpy as np
import cvxopt


class SVMClassifier:

    def __init__(self, kernel='linear', C=None, TOL=1e-7, show_progress=False, **kwargs):
        self.fit_done = False
        self.C = C
        self.TOL = TOL
        self.show_progress = show_progress
        if kernel == 'linear':
            self.kernel_func = self.__kernel_linear
        elif kernel == 'poly':
            poly_c = kwargs.get('poly_c', 0)
            poly_d = kwargs.get('poly_d', 2)
            self.kernel_func = lambda x,y: self.__kernel_poly(x, y, poly_c, poly_d)
        elif kernel == 'rbf':
            rbf_sigma = kwargs.get('rbf_sigma', 5)
            self.kernel_func = lambda x,y: self.__kernel_rbf(x, y, rbf_sigma)

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

        self.b = 0
        for a_i, sv_i, svt_i in zip(self.lambda_sv, self.sv, self.svt):
            tmp = sum(a_j * svt_j * self.kernel_func(sv_i, sv_j)
                      for a_j, sv_j, svt_j in zip(self.lambda_sv, self.sv, self.svt))
            self.b += svt_i - tmp
        self.b /= sv_num

        return self

    def predict(self, X):
        y_predict = np.array([sum(a_i * svt_i * self.kernel_func(sv_i, x)
                                  for a_i, sv_i, svt_i in zip(self.lambda_sv, self.sv, self.svt))
                              for x in X])

        return np.array(list(map(lambda x: self.label_map[x],
                                 np.sign(y_predict + self.b)))
                        ),\
               y_predict + self.b

    def __generate_lambda(self, X, y):
        n, features = X.shape

        # http://cvxopt.org/userguide/coneprog.html#quadratic-programming
        # need to maximize L(lambda) w.r.t. lambda --> minimize -L(lambda)
        P = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ker_val = self.kernel_func(X[i,:], X[j,:])
                P[i,j] = y[i]*y[j]*ker_val

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


    def __kernel_linear(self, x, y):
        return np.dot(x, y)

    def __kernel_poly(self, x, y, c, d):
        return np.power((np.dot(x, y) + c), d)

    def __kernel_rbf(self, x, y, sigma):
        # x = x.reshape(-1, )
        # y = y.reshape(-1, )
        vec_diff = x - y
        norm_squared = np.power(np.linalg.norm(vec_diff), 2)
        return np.exp(-norm_squared/(2*np.power(sigma, 2)))