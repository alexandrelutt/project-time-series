import numpy as np

from source_code import utils
from source_code.twi_omp import TWI_OMP, align_on

class TWI_kSVD():
    def __init__(self, max_iter=5, verbose=False, init_dict=None):
        self.max_iter = max_iter
        self.is_fit = False
        self.verbose = verbose
        self.init_dict = init_dict
        if not (self.init_dict is None):
            self._D = init_dict
            self.is_fit = True
    
    def init_dictionnary(self, X):
        return utils.create_dictionary(X, n_classes=3, num_atoms_per_class=10)

    def _encode_dataset(self, X, sparsity):
        self.A, self.deltas, self.all_Ds = [], [], []
        for i in range(self.N):
            alpha, delta, Ds = TWI_OMP().encode(X[i], self._D, sparsity)
            self.A.append(alpha)
            self.deltas.append(delta)
            self.all_Ds.append(Ds)
        self.A = np.array(self.A).reshape(self.N, self.K)
        self.all_Ds = np.array(self.all_Ds).reshape(self.N, self.K, -1)

    def _get_E(self, X, k, w_k):
        E_k = np.zeros((len(self._D[k]), len(w_k)))
        _E_k = np.zeros((self.n, len(w_k)))
        for j, i in enumerate(w_k):
            e_i = X[i] - np.delete(self.all_Ds[i], k, axis=0).T @ np.delete(self.A[i], k, axis=0)
            _E_k[:, j] = e_i

            a = (self.deltas[i][k].T @ e_i).reshape(-1, 1)
            
            b = (self.deltas[i][k].T @ self.all_Ds[i][k]).reshape(-1, 1)
            
            c = (self._D[k]).reshape(-1, 1)
            
            phi_e_i = utils.rotation(a, b, c)
            E_k[:, j] = phi_e_i.reshape(-1)
        return E_k, _E_k
    
    def _update_assignments(self, u_1, k, w_k, _E_k):
        for j, i in enumerate(w_k):
            a = u_1.reshape(-1, 1)
            b = self._D[k].reshape(-1, 1)

            c = (self.deltas[i][k].T @ self.all_Ds[i][k]).reshape(-1, 1)
            
            gamma_i_u = (self.deltas[i][k] @ utils.rotation(a, b, c)).reshape(-1)

            self.all_Ds[i][k] = gamma_i_u
            self.A[i, k] = _E_k[:, j].T @ gamma_i_u
            self.A[i, k] = self.A[i, k]/np.linalg.norm(gamma_i_u)

    def fit(self, X, sparsity=10):
        if self.verbose:
            print(f'Training TWI-kSVD model...')
        self._D = self.init_dictionnary(X)

        self.n = len(X[0])
        self.N = len(X)
        self.K = len(self._D)

        for t in range(self.max_iter):
            if self.verbose:
                print(f'  Encoding dataset (iteration {t+1})...')
            self._encode_dataset(X, sparsity)

            if self.verbose:
                print('    Dataset succesfully encoded.')
                print('  Updating atoms...')

            for k in range(self.K):
                w_k = np.where(np.abs(self.A[:, k]) > 1e-7)[0]
                if len(w_k):
                    E_k, _E_k = self._get_E(X, k, w_k)                 
                    U_k, _, _ = np.linalg.svd(E_k)
                    u_1 = U_k[0, :]
                    self._update_assignments(u_1, k, w_k, _E_k)
                    self._D[k] = u_1
            if self.verbose:
                print('    Atoms succesfully updated.\n')

        if self.verbose:
            print(f'TWI-kSVD model has been succesfully trained.\n')
        self.is_fit = True
        return self

    def reconstruct(self, x, sparsity=10):
        if not self.is_fit:
            raise Exception('Model not fit yet.')
        alpha, _, array = TWI_OMP().encode(x, self._D, sparsity)
        reconstructed = array @ alpha
        return reconstructed.reshape(-1)
    
    def fit_transform(self, X, sparsity=10):
        self.fit(X)
        reconstruction = np.array([self.reconstruct(x, sparsity) for x in X])
        return reconstruction