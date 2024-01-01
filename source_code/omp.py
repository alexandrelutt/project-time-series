import numpy as np
import scipy

def omp_single(x, D, tau):
    residual = x
    idx = []

    while len(idx) < tau:
        lam = np.argmax(np.abs(np.dot(residual, D)))
        idx.append(lam)
        alpha, _, _, _ = scipy.linalg.lstsq(D[:, idx], x)
        residual = x - np.dot(D[:, idx], alpha)
        
    large_alpha = np.zeros(D.shape[1])
    large_alpha[idx] = alpha
    return large_alpha

def omp(X, D, tau):
    """
    Perform Orthogonal Matching Pursuit (OMP) on a given dataset.

    Parameters:
    D (numpy.ndarray): Dictionary matrix of shape (len(signal), len(basis)) containing the basis vectors as columns.
    X (numpy.ndarray): Signal matrix of shape (n_signal, len(signal)) to be represented.
    tau (int): Sparsity level, i.e. the maximum number of non-zero coefficients in the representation of each signal.

    Returns:
    gammas (numpy.ndarray): Array of shape (n_signal, len(basis)) of representations for each signal in X. Each row corresponds to a signal, and each column corresponds to a basis in D.
    """
    alphas = np.apply_along_axis(func1d=omp_single, axis=1, arr=X, D=D, tau=tau)
    return alphas