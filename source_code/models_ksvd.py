import numpy as np

from source_code import utils
from source_code.omp import omp, omp_single
    
class kSVD():
    def __init__(self, n_classes, max_iter=20, tol=1e-6, init_dict=None):
        self.max_iter = max_iter
        self.tol = tol
        self.is_fit = False
        self.init_dict = init_dict
        self.n_classes = n_classes
        if not (self.init_dict is None):
            self.D_list = init_dict
            self.is_fit = True

    def _update_dict(self, X, D, alpha):
        for k in range(D.shape[1]):
            I = np.nonzero(alpha[:, k])[0]
            if len(I) == 0:
                continue

            E = X[:, I] - D @ alpha[I, :].T
            U, S, V = np.linalg.svd(E, full_matrices=False)
            D[:, k] = U[:, 0]
            alpha[I, k] = S[0] * U[0, :]
        return D, alpha

    def fit(self, X, y, sparsity=10):
        X_classes = []
        self.D_list = []
        for _ in np.unique(y):
            X_class = X[:, y == _]
            X_classes.append(X_class)

        for c, X_class in enumerate(X_classes):
            D = utils.init_dictionnary(X_class, n_classes=self.n_classes)
            for _ in range(self.max_iter):
                alpha = omp(X_class.T, D, sparsity)
                D, alpha = self._update_dict(X_class, D, alpha)

                error = np.linalg.norm(X_class - D @ alpha.T)/np.linalg.norm(X_class)
                if error < self.tol:
                    break

            self.D_list.append(D)

        self.is_fit = True
        return self

    def get_best_candidate(self, x, sparsity):
        best_err = np.inf
        best_c = None
        for c in range(self.n_classes):
            alphas = omp_single(x.T, self.D_list[c], sparsity)
            reconstruction = (self.D_list[c] @ alphas.T)
            err = np.linalg.norm(x - reconstruction)
            if err < best_err:
                best_err = err
                best_c = c
                best_alphas = alphas
        return best_c, best_alphas
    
    def reconstruct(self, x, sparsity=10):
        if not self.is_fit:
            raise Exception('Model not fit yet.')
        
        best_c, best_alphas = self.get_best_candidate(x, sparsity=sparsity)
        reconstruction = (self.D_list[best_c] @ best_alphas.T)
        return reconstruction, best_c

class kSVD_2D():
    def __init__(self, n_classes, max_iter=20, init_dicts=None):
        self.max_iter = max_iter
        self.is_fit = False
        self.init_dicts = init_dicts
        self.n_classes = n_classes
        self.D_list_X, self.D_list_Y = None, None
        if not (self.init_dicts is None):
            self.D_list_X = self.init_dicts[0]
            self.D_list_Y = self.init_dicts[1]
            self.is_fit = True
        
        self.model_1D_X = kSVD(n_classes=self.n_classes, max_iter=self.max_iter, init_dict=self.D_list_X)
        self.model_1D_Y = kSVD(n_classes=self.n_classes, max_iter=self.max_iter, init_dict=self.D_list_Y)

    def fit(self, X, Y, labels, sparsity=10):
        self.model_1D_X.fit(X, labels, sparsity=sparsity)
        self.model_1D_Y.fit(Y, labels, sparsity=sparsity)
        self.D_list_X = self.model_1D_X.D_list
        self.D_list_Y = self.model_1D_Y.D_list
        self.is_fit = True
        return self

    def get_best_candidate(self, x, y, sparsity):
        best_err = np.inf
        best_c = None
        for c in range(self.n_classes):
            alphas_x = omp_single(x.T, self.D_list_X[c], sparsity)
            alphas_y = omp_single(y.T, self.D_list_Y[c], sparsity)
            reconstruction_x = (self.D_list_X[c] @ alphas_x.T)
            reconstruction_y = (self.D_list_Y[c] @ alphas_y.T)
            original_signal = np.array([x, y])
            reconstruction_signal = np.array([reconstruction_x, reconstruction_y])
            err = np.linalg.norm(original_signal - reconstruction_signal)

            if err < best_err:
                best_err = err
                best_c = c
                best_alphas_x = alphas_x
                best_alphas_y = alphas_y

        return best_c, best_alphas_x, best_alphas_y

    def reconstruct(self, x, y, sparsity=10):
        if not self.is_fit:
            raise Exception('Model not fit yet.')
        
        best_c, best_alphas_x, best_alphas_y = self.get_best_candidate(x, y, sparsity=sparsity)
        reconstruction_x = (self.D_list_X[best_c] @ best_alphas_x.T)
        reconstruction_y = (self.D_list_Y[best_c] @ best_alphas_y.T)
        return reconstruction_x, reconstruction_y, best_c
