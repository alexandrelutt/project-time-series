import numpy as np
import matplotlib.pyplot as plt

from source_code import utils
from source_code.twi_omp import TWI_OMP

class TWI_kSVD():
    def __init__(self, max_iter=10, n_classes=10, init_D_list=None):
        self.max_iter = max_iter
        self.n_classes = n_classes
        self.n_atoms = 10
        self.is_fit = False
        self.init_D_list = init_D_list
        if not (self.init_D_list is None):
            self.D_list = init_D_list
            self.classif_D = [d for D in self.D_list for d in D]
            self.is_fit = True

    def _encode_dataset(self, X, sparsity):
        self.A, self.deltas, self.all_Ds = [], [], []
        for i in range(len(X)):
            alpha, delta, Ds = TWI_OMP().encode(X[i], self._D, sparsity)
            ## alpha = sparse asignements
            ## deltas = list of matrices to project onto D
            ## Ds = array of projected atoms )
            self.A.append(alpha)
            self.deltas.append(delta)
            self.all_Ds.append(Ds)

    def _get_E(self, X, k, w_k):
        _phi_E_is = []
        _E_is = [0]*len(self.A)
        for i in w_k:  
            D_array = self.all_Ds[i].copy()
            D_array_k = D_array[:, k].reshape(-1, 1)
            D_array = np.delete(D_array, k, axis=1)

            delta_k = self.deltas[i][k].copy()
            d_k = self._D[k].copy()

            current_A = self.A[i].copy()
            current_A = np.delete(current_A, k, axis=0)

            e_i = (X[i] - (D_array @ current_A).reshape(-1))
            _E_is[i] = e_i

            a = (delta_k.T @ e_i).reshape(-1, 1)
            b = (delta_k.T @ D_array_k).reshape(-1, 1)
            c = d_k.reshape(-1, 1)
            
            phi_e_i = utils.rotation(a, b, c)

            _phi_E_is.append(phi_e_i.reshape(-1))

        _phi_E_is = np.array(_phi_E_is).T
        return _phi_E_is, _E_is
    
    def _update_assignments(self, u_1, k, w_k, _E_is):
        for i in w_k:
            a = u_1.reshape(-1, 1)
            b = self._D[k].reshape(-1, 1)
            c = (self.deltas[i][k].T @ self.all_Ds[i][:, k]).reshape(-1, 1)

            r = utils.rotation(a, b, c)            
            gamma_i_u = (self.deltas[i][k] @ r).reshape(-1)

            self.all_Ds[i][:, k] = gamma_i_u
            new_coef = np.dot(_E_is[i].T, gamma_i_u)/np.linalg.norm(gamma_i_u)
            self.A[i][k] = new_coef

    def fit(self, large_X, labels, sparsity, dataset_name):
        X_classes = []
        self.D_list = []
        for label in np.unique(labels):
            X = [large_X[i] for i in range(len(labels)) if labels[i] == label]
            X_classes.append(X)

        for c, X in enumerate(X_classes):
            self._D = utils.init_list_dictionnary(X, n_atoms=self.n_atoms)

            errors = []
            for t in range(self.max_iter):
                self._encode_dataset(X, sparsity)

                # if c == 0 and sparsity == 2 and not '_Y' in dataset_name:
                #     error = 0
                #     for i in range(len(X)):
                #         x = X[i]
                #         alphas, _, Ds = TWI_OMP().encode(x, self._D, sparsity)
                #         reconstructed = (Ds @ alphas).reshape(-1)
                #         error += np.linalg.norm(x - reconstructed)/np.linalg.norm(x)
                #     errors.append(error/len(X))

                for k in range(len(self._D)):
                    w_k = [i for i, A in enumerate(self.A) if np.abs(A[k]) > 1e-7]
                    if len(w_k):
                        _phi_E_is, _E_is = self._get_E(X, k, w_k)   
                        U_k, _, _ = np.linalg.svd(_phi_E_is)
                        u_1 = U_k[:, 0]
                        
                        self._D[k] = u_1.reshape(-1)
                        self._update_assignments(u_1, k, w_k, _E_is)
                    
            # if c == 0 and sparsity == 2 and not '_Y' in dataset_name:
            #     plt.plot(errors)
            #     plt.title(f'Reconstruction error for TWI-kSVD trained on {dataset_name} (class {c}, sparsity {sparsity})')
            #     plt.xlabel('Iteration')
            #     plt.ylabel('Reconstruction error')
            #     plt.savefig(f'figures/loss_TWI_kSVD_spars_{sparsity}_class_{c}_{dataset_name}.png')
            #     plt.close()

            self.D_list.append(self._D)

        self.classif_D = [d for D in self.D_list for d in D]
        self.is_fit = True
        return self

    def get_prediction(self, x, alphas, Ds):
        best_err = np.inf
        for c in range(self.n_classes):
            local_alpha = alphas[c*self.n_atoms:(c+1)*self.n_atoms]
            D = Ds[:, c*self.n_atoms:(c+1)*self.n_atoms]
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
            alpha, _, Ds = TWI_OMP().encode(x, D, sparsity)
            small_sample = (Ds @ alpha).reshape(-1)
            err = np.linalg.norm(small_sample - x)
            if err < best_err:
                best_err = err
                best_c = c
        return best_c

    def reconstruct(self, x, sparsity, only_l2=False):
        if not self.is_fit:
            raise Exception('Model not fit yet.')

        alphas, _, Ds = TWI_OMP().encode(x, self.classif_D, sparsity)
        reconstructed = (Ds @ alphas).reshape(-1)
        if not only_l2:
            y_pred = self.get_prediction(x, alphas, Ds)
            my_y_pred = self.get_my_prediction(x, sparsity)
        else:
            y_pred, my_y_pred = None, None
        return reconstructed, y_pred, my_y_pred

class TWI_kSVD_2D():
    def __init__(self, max_iter=10, n_classes=10, init_D_list=None):
        self.max_iter = max_iter
        self.n_classes = n_classes
        self.n_atoms = 10
        self.is_fit = False
        self.D_list_X, self.D_list_Y = None, None
        self.init_D_list = init_D_list
        if not (self.init_D_list is None):
            self.D_list_X = self.init_D_list[0]
            self.D_list_Y = self.init_D_list[1]
            self.is_fit = True

        self.model_1D_X = TWI_kSVD(n_classes=self.n_classes, max_iter=self.max_iter, init_D_list=self.D_list_X)
        self.model_1D_Y = TWI_kSVD(n_classes=self.n_classes, max_iter=self.max_iter, init_D_list=self.D_list_Y)

    def fit(self, large_X, large_Y, labels, sparsity, dataset_name):
        self.model_1D_X.fit(large_X, labels, sparsity, dataset_name + '_X')
        self.model_1D_Y.fit(large_Y, labels, sparsity, dataset_name + '_Y')
        self.D_list_X = self.model_1D_X.D_list
        self.D_list_Y = self.model_1D_Y.D_list
        self.is_fit = True
        return self

    def get_prediction(self, x, y, alphas_x, alphas_y, Ds_x, Ds_y):
        best_err = np.inf
        for c in range(self.n_classes):
            local_alpha_x = alphas_x[c*self.n_atoms:(c+1)*self.n_atoms]
            local_alpha_y = alphas_y[c*self.n_atoms:(c+1)*self.n_atoms]

            local_Ds_x = Ds_x[:, c*self.n_atoms:(c+1)*self.n_atoms]
            local_Ds_y = Ds_y[:, c*self.n_atoms:(c+1)*self.n_atoms]

            small_sample_x = (local_Ds_x @ local_alpha_x).reshape(-1)
            small_sample_y = (local_Ds_y @ local_alpha_y).reshape(-1)

            err = np.linalg.norm(small_sample_x - x) + np.linalg.norm(small_sample_y - y)
            if err < best_err:
                best_err = err
                best_c = c
        return best_c

    def get_my_prediction(self, x, y, sparsity):
        best_err = np.inf
        for c in range(self.n_classes):
            D_x = self.D_list_X[c]
            D_y = self.D_list_Y[c]
            alpha_x, _, Ds_x = TWI_OMP().encode(x, D_x, sparsity)
            alpha_y, _, Ds_y = TWI_OMP().encode(y, D_y, sparsity)
            small_sample_x = (Ds_x @ alpha_x).reshape(-1)
            small_sample_y = (Ds_y @ alpha_y).reshape(-1)
            err = np.linalg.norm(small_sample_x - x) + np.linalg.norm(small_sample_y - y)
            if err < best_err:
                best_err = err
                best_c = c
        return best_c

    def reconstruct(self, x, y, sparsity, only_l2=False):
        if not self.is_fit:
            raise Exception('Model not fit yet.')

        alphas_x, _, Ds_x = TWI_OMP().encode(x, self.model_1D_X.classif_D, sparsity)
        reconstructed_x = (Ds_x @ alphas_x).reshape(-1)
        alphas_y, _, Ds_y = TWI_OMP().encode(y, self.model_1D_Y.classif_D, sparsity)
        reconstructed_y = (Ds_y @ alphas_y).reshape(-1)

        if not only_l2:
            y_pred = self.get_prediction(x, y, alphas_x, alphas_y, Ds_x, Ds_y)
            my_y_pred = self.get_my_prediction(x, y, sparsity)

        else:
            y_pred, my_y_pred = None, None
            
        return reconstructed_x, reconstructed_y, y_pred, my_y_pred