import numpy as np
import matplotlib.pyplot as plt

from source_code import utils
from source_code.omp import omp, omp_single
    
class kSVD():
    def __init__(self, n_classes, max_iter=10, init_D_list=None):
        self.max_iter = max_iter
        self.is_fit = False
        self.init_D_list = init_D_list
        self.n_classes = n_classes
        self.n_atoms = 10
        if not (self.init_D_list is None):
            self.D_list = init_D_list
            self.classif_D = np.concatenate(self.D_list, axis=1)
            self.is_fit = True

    def _update_dict(self, X, D, alpha, k):
        I = np.where(np.abs(alpha[:, k]) > 1e-7)[0]
        if len(I) == 0:
            return D, alpha
                
        E = X - np.delete(D, k, axis=1) @ np.delete(alpha, k, axis=1).T  
        E = E[:, I]
        U, S, V = np.linalg.svd(E, full_matrices=True)
        D[:, k] = U[:, 0]
        alpha[I, k] = S[0] * V[0, :]
        return D, alpha

    def fit(self, X, y, sparsity, dataset_name):
        X_classes = []
        self.D_list = []
        for _ in np.unique(y):
            X_class = X[:, y == _]
            X_classes.append(X_class)

        for c, X_class in enumerate(X_classes):
            D = utils.init_dictionnary(X_class, n_atoms=self.n_atoms)
            errors = []
            for _ in range(self.max_iter):
                alpha = omp(X_class.T, D, sparsity)
                for k in range(D.shape[1]):
                    D, alpha = self._update_dict(X_class, D, alpha, k)

                if c == 0 and sparsity == 2 and not '_Y' in dataset_name:
                    error = 0
                    for i in range(X.shape[1]):
                        x = X[:, i]
                        alphas = omp_single(x, D, sparsity)
                        error += np.linalg.norm(x - (D @ alphas).reshape(-1))/np.linalg.norm(X[:, i])
                    errors.append(error/X.shape[1])

            if c == 0 and sparsity == 2 and not '_Y' in dataset_name:
                plt.plot(errors)
                plt.title(f'Reconstruction error for kSVD trained on {dataset_name} (class {c}, sparsity {sparsity})')
                plt.xlabel('Iteration')
                plt.ylabel('Reconstruction error')
                plt.savefig(f'figures/loss_kSVD_spars_{sparsity}_class_{c}_{dataset_name}.png')
                plt.close()

            self.D_list.append(D)

        self.classif_D = np.concatenate(self.D_list, axis=1)
        self.is_fit = True
        return self

    def get_prediction(self, x, alphas):
        best_err = np.inf
        for c in range(self.n_classes):
            D = self.D_list[c]
            local_alpha = alphas[c*self.n_atoms:(c+1)*self.n_atoms]
            small_sample = (D @ local_alpha).reshape(-1)
            err = np.linalg.norm(small_sample - x)
            if err < best_err:
                best_err = err
                best_c = c
        return best_c

    def get_my_prediction(self, x, sparsity):
        best_err = np.inf
        for c in range(self.n_classes):
            D = self.D_list[c]
            alpha = omp_single(x, D, sparsity)
            small_sample = (D @ alpha).reshape(-1)
            err = np.linalg.norm(small_sample - x)
            if err < best_err:
                best_err = err
                best_c = c
        return best_c
    
    def reconstruct(self, x, sparsity, only_l2=False):
        if not self.is_fit:
            raise Exception('Model not fit yet.')
        alphas = omp_single(x, self.classif_D, sparsity)
        reconstruction = self.classif_D @ alphas.T
        if not only_l2:
            y_pred = self.get_prediction(x, alphas)
            my_y_pred = self.get_my_prediction(x, sparsity)
        else:
            y_pred, my_y_pred = None, None
        return reconstruction, y_pred, my_y_pred

class kSVD_2D():
    def __init__(self, n_classes, max_iter=10, init_D_list=None):
        self.max_iter = max_iter
        self.is_fit = False
        self.n_classes = n_classes
        self.n_atoms = 10
        self.D_list_X, self.D_list_Y = None, None
        if not (init_D_list is None):
            self.D_list_X = init_D_list[0]
            self.D_list_Y = init_D_list[1]
            self.is_fit = True
        
        self.model_1D_X = kSVD(n_classes=self.n_classes, max_iter=self.max_iter, init_D_list=self.D_list_X)
        self.model_1D_Y = kSVD(n_classes=self.n_classes, max_iter=self.max_iter, init_D_list=self.D_list_Y)

    def fit(self, X, Y, labels, sparsity, dataset_name):
        self.model_1D_X.fit(X, labels, sparsity=sparsity, dataset_name=dataset_name + '_X')
        self.model_1D_Y.fit(Y, labels, sparsity=sparsity, dataset_name=dataset_name + '_Y')
        self.D_list_X = self.model_1D_X.D_list
        self.D_list_Y = self.model_1D_Y.D_list
        self.is_fit = True
        return self

    def get_prediction(self, x, y, alphas_x, alphas_y):
        best_err = np.inf
        for c in range(self.n_classes):
            D_x = self.D_list_X[c]
            alpha_x = alphas_x[c*self.n_atoms:(c+1)*self.n_atoms]
            small_sample_x = (D_x @ alpha_x).reshape(-1)

            D_y = self.D_list_Y[c]
            alpha_y = alphas_y[c*self.n_atoms:(c+1)*self.n_atoms]
            small_sample_y = (D_y @ alpha_y).reshape(-1)

            err = np.linalg.norm(small_sample_x - x) + np.linalg.norm(small_sample_y - y)

            if err < best_err:
                best_err = err
                best_c = c
        return best_c

    def get_my_prediction(self, x, y, sparsity):
        best_err = np.inf
        for c in range(self.n_classes):
            D_x = self.D_list_X[c]
            alpha_x = omp_single(x, D_x, sparsity)
            small_sample_x = (D_x @ alpha_x).reshape(-1)

            D_y = self.D_list_Y[c]
            alpha_y = omp_single(y, D_y, sparsity)
            small_sample_y = (D_y @ alpha_y).reshape(-1)

            err = np.linalg.norm(small_sample_x - x) + np.linalg.norm(small_sample_y - y)

            if err < best_err:
                best_err = err
                best_c = c
        return best_c

    def reconstruct(self, x, y, sparsity, only_l2=False):
        if not self.is_fit:
            raise Exception('Model not fit yet.')
        
        alphas_x = omp_single(x, self.model_1D_X.classif_D, sparsity)
        reconstruction_x = self.model_1D_X.classif_D @ alphas_x.T
        alphas_y = omp_single(y, self.model_1D_Y.classif_D, sparsity)
        reconstruction_y = self.model_1D_Y.classif_D @ alphas_y.T
        if not only_l2:
            y_pred = self.get_prediction(x, y, alphas_x, alphas_y)
            my_y_pred = self.get_my_prediction(x, y, sparsity)
        else:
            y_pred, my_y_pred = None, None
        return reconstruction_x, reconstruction_y, y_pred, my_y_pred
