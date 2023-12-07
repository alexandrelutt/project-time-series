import numpy as np
    
class OMP():
    def get_best_atom_idx(self):
        dot_products = np.abs(np.dot(self.D.T, self.residual)/np.linalg.norm(self.D, axis=0).reshape(-1, 1))
        best_idx = np.argmax(dot_products)
        return best_idx

    def get_alpha(self, x):
        alpha = np.linalg.lstsq(self.D[:, self.support], x, rcond=None)[0]
        return alpha

    def encode(self, x, D, sparsity):
        self.D = D
        m, n = self.D.shape
        self.residual = x.copy()
        self.support = []
        self.alpha = np.zeros((n, 1))
        while len(self.support) < sparsity:
            self.current_atom_idx = self.get_best_atom_idx()
            self.support.append(self.current_atom_idx)
            self.current_alpha = self.get_alpha(x)
            self.alpha[self.support] = self.current_alpha
            self.residual = x - np.dot(self.D[:, self.support], self.current_alpha)
            if np.linalg.norm(self.residual) < 1e-6:
                break
        return self.alpha, np.dot(D, self.alpha)
    
    def encode_batch(self, X, D, sparsity):
        alphas = []
        X = X.reshape(X.shape[0], X.shape[1], 1)
        for x in X:
            alphas.append(self.encode(x, D, sparsity)[0])
        return np.array(alphas).reshape(len(X), -1)