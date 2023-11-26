import numpy as np

class OMP():
    def get_best_atom_idx(self):
        dot_products = np.abs(np.dot(self.D.T, self.residual)/np.linalg.norm(self.D, axis=0).reshape(-1, 1))
        best_idx = np.argmax(dot_products)
        return best_idx

    def get_alpha(self, x):
        alpha = np.linalg.lstsq(self.D[:, self.support], x, rcond=None)[0]
        return alpha

    def encode(self, x, D, tau):
        self.D = D
        m, n = self.D.shape
        self.residual = x.copy()
        self.support = []
        self.alpha = np.zeros((n, 1))
        while len(self.support) < tau:
            self.current_atom_idx = self.get_best_atom_idx()
            self.support.append(self.current_atom_idx)
            self.current_alpha = self.get_alpha(x)
            self.alpha[self.support] = self.current_alpha
            self.residual = x - np.dot(self.D[:, self.support], self.current_alpha)
            if np.linalg.norm(self.residual) < 1e-6:
                break
        return self.alpha, np.dot(D, self.alpha)
    
    def encode_batch(self, X, D, tau):
        alphas = []
        X = X.reshape(X.shape[0], X.shape[1], 1)
        for x in X:
            alphas.append(OMP().encode(x, D, tau)[0])
        return np.array(alphas).reshape(len(X), -1)
    
class kSVD(object):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, X, D, alpha):
        for j in range(self.n_components):
            I = alpha[:, j] > 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            g = alpha[I, j].T
            r = X[I, :] - alpha[I, :].dot(D)
            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)
            D[j, :] = d
            alpha[I, j] = g.T
        return D, alpha

    def initialize_dict(self, X):
        D = np.random.randn(self.n_components, X.shape[1])
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _encode(self, D, X, sparsity):
        alphas = OMP().encode_batch(X, D.T, sparsity)
        return alphas

    def fit(self, X, sparsity):
        D = self.initialize_dict(X)
        for _ in range(self.max_iter):
            alpha = self._encode(D, X, sparsity)
            error = np.linalg.norm(X - alpha.dot(D))
            if error < self.tol:
                break
            D, alpha = self._update_dict(X, D, alpha)

        self.components_ = D
        return self

    def encode(self, X, sparsity):
        return self._encode(self.components_, X, sparsity)