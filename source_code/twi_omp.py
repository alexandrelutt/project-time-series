import numpy as np

def f(list):
    if not list[0]:
        return 0
    return list[0]/np.sqrt(list[1]*list[2])

def vect_f(matrix):
    output_matrix = np.zeros(matrix[:, :, 0].shape)
    output_matrix[np.sqrt(matrix[:, :, 1])*np.sqrt(matrix[:, :, 2]) > 0] = matrix[:, :, 0][np.sqrt(matrix[:, :, 1])*np.sqrt(matrix[:, :, 2]) > 0]/(np.sqrt(matrix[:, :, 1])*np.sqrt(matrix[:, :, 2]))[np.sqrt(matrix[:, :, 1])*np.sqrt(matrix[:, :, 2]) > 0]
    return output_matrix

def argmax(list, funct):
    id = np.argmax([funct(l) for l in list])
    return list[id]

def get_next_cell(i, j, id):
    if id == 0:
        return (i, j-1)
    elif id == 1:
        return (i-1, j)
    else:
        return (i-1, j-1)

def compute_matrix(x, y):
    Tx, Ty = len(x), len(y)
    
    M = np.zeros((Tx, Ty, 3))
    
    M[0, 0] = np.array([x[0]*y[0], x[0]**2, y[0]**2])

    for i in range(1, Tx):
        M[i, 0] = M[i-1, 0] + np.array([x[i]*y[0], x[i]**2, y[0]**2])
    for j in range(1, Ty):
        M[0, j] = M[0, j-1] + np.array([x[0]*y[j], x[0]**2, y[j]**2])
    
    for i in range(1, Tx):
        for j in range(1, Ty):
            candidates = [
                M[i, j-1] + np.array([x[i]*y[j], x[i]**2, y[j]**2]),
                M[i-1, j] + np.array([x[i]*y[j], x[i]**2, y[j]**2]),
                M[i-1, j-1] + np.array([x[i-1]*y[j-1], x[i-1]**2, y[j-1]**2]),
            ]
            M_ij = argmax(candidates, f)
            M[i, j] = M_ij
    return M

def get_alignements(costw_matrix):
    Tx, Ty = costw_matrix.shape
    alignements = np.zeros((Tx, Ty))
    alignements[Tx-1, Ty-1 ] = 1

    i, j = Tx-1, Ty-1 
    while (i, j) != (0, 0):
        if i == 0:
            (i, j) = (i, j-1)
        elif j == 0:
            (i, j) = (i-1, j)
        else:
            id = np.argmax([costw_matrix[i-1, j-1], costw_matrix[i-1, j], costw_matrix[i, j-1]])
            if id == 0:
                (i, j) = (i-1, j-1)
            if id == 1:
                (i, j) = (i-1, j)
            if id == 2:
                (i, j) = (i, j-1)
        alignements[i, j] = 1

    return alignements

# def align_on(alignement, s2):
#     new_s2 = np.zeros(alignement.shape[0])
#     for i in range(alignement.shape[0]):
#         j = np.argmax(alignement[i, :])
#         new_s2[i] = s2[j]
#     return new_s2

def align_on(alignement, s2):
    indices = np.argmax(alignement, axis=1)
    new_s2 = s2[indices]
    return new_s2

def costw(x, y):
    M = compute_matrix(x, y)
    costw_matrix = vect_f(M)
    alignements = get_alignements(costw_matrix)
    return costw_matrix[-1, -1], alignements

class TWI_OMP():
    def get_best_candidate(self):
        best_cost = -np.inf
        for j in range(len(self.D)):
            if j not in self.support:
                candidate = self.D[j]
                cost, alignements = costw(self.residual, candidate)
                aligned_candidate = align_on(alignements, candidate)
                if cost > best_cost:
                    best_cost = cost
                    best_j = j
                    best_alignement = alignements
                    best_aligned_candidate = aligned_candidate
        return best_aligned_candidate, best_j, best_alignement

    def get_alpha(self, x):
        alpha = np.linalg.lstsq(self.D_array, x.reshape(-1, 1), rcond=None)[0]
        return alpha

    def encode(self, x, D, sparsity):
        self.D = D.copy()
        n = len(self.D)
        self.residual = x.copy()
        self.support = []
        self.alpha = np.zeros((n, 1))
        self.deltas = [np.zeros((len(x), len(d))) for d in D]
        self.D_array = np.zeros((len(self.residual), len(self.D)))
        while len(self.support) < sparsity:
            best_aligned_candidate, best_candidate_id, best_alignement = self.get_best_candidate()
            self.deltas[best_candidate_id] = best_alignement
        
            self.support.append(best_candidate_id)
            self.D_array[:, best_candidate_id] = best_aligned_candidate.reshape(-1)

            self.alpha = self.get_alpha(x).reshape(-1, 1)
            self.residual = (x.reshape(-1, 1) - self.D_array @ self.alpha).reshape(-1)

            err = np.linalg.norm(self.residual)
            if err < 1e-6:
                break

        return self.alpha, self.deltas, self.D_array
    
    def encode_batch(self, X, D, sparsity):
        alphas, deltas = [], []
        X = X.reshape(X.shape[0], X.shape[1], 1)
        for x in X:
            alpha, delta, _, _ = self.encode(x[:, 0], D, sparsity)
            alphas.append(alpha)
            deltas.append(delta)
        return np.array(alphas).reshape(len(X), -1), deltas
    