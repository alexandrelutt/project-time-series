import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import time

from source_code.models import kSVD, TWI_kSVD

np.random.seed(0)

def load_dataset(dataset_name):
    if dataset_name == 'BME':
        train_data = arff.loadarff(f'datasets/{dataset_name}/{dataset_name}_TRAIN.arff')[0]
        test_data = arff.loadarff(f'datasets/{dataset_name}/{dataset_name}_TEST.arff')[0]
        
    else:
        raise NotImplementedError
    
    return train_data, test_data

def preprocess(dataset, serie_length):
    matrix = np.zeros((dataset.shape[0], serie_length))
    labels = np.zeros((dataset.shape[0], 1))

    for i, serie in enumerate(dataset):
        label = serie[-1]
        label = np.frombuffer(label, dtype=np.uint8)[0] - 49

        serie = np.array(list(serie)[:-1])
        matrix[i, :] = serie
        labels[i] = label
    
    return matrix, labels

def plot_random_example(train_matrix, alphas, D):
    i = np.random.randint(0, train_matrix.shape[0])
    sample = train_matrix[i, :].reshape(-1, 1)
    reconstructed_sample = alphas[i, :] @ D

    plt.figure(figsize=(10, 5))
    plt.plot(sample, color='b', label='Original signal')
    plt.plot(reconstructed_sample, color='r', label='Reconstructed signal', linestyle='--')
    plt.title('Original signal vs reconstructed signal (kSVD)')
    plt.legend()
    plt.show()

def rotation(a, b, c):
    len = a.shape[0]
    theta = np.arccos((b.T @ c)/(np.linalg.norm(b) * np.linalg.norm(c)))[0, 0]
    u = b / np.linalg.norm(b)
    v = (c - (u.T @ c)*u)/np.linalg.norm((c - (u.T @ c)*u))
    
    R_theta = np.hstack([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).reshape(2, 2)
    R = np.eye(len) - u @ u.T - v @ v.T + np.hstack([u, v]) @ R_theta @ np.hstack([u, v]).T
    
    ar = R @ a
    
    return ar

def create_dictionary(matrix, n_classes, num_atoms_per_class=10, fixed_size=False):
    dictionary_list = []
    signal_size = matrix.shape[1]

    for i in range(n_classes):
        class_matrix = matrix[i*num_atoms_per_class:(i+1)*num_atoms_per_class, :]
        ids = np.random.randint(0, len(class_matrix), num_atoms_per_class)
        if fixed_size:
            for j in range(num_atoms_per_class):
                dictionary_list.append(np.random.random(signal_size))
        else:
            lengths = np.random.randint(int(0.4*signal_size), int(0.6*signal_size), num_atoms_per_class)
            poss = np.random.randint(0, signal_size-max(lengths), num_atoms_per_class)
            for j in range(num_atoms_per_class):
                dictionary_list.append(class_matrix[ids[j], poss[j]:poss[j]+lengths[j]].reshape(-1))
    if fixed_size:
        dictionary_list = np.array(dictionary_list)
    return dictionary_list

def show_sample(X, model, model_name, id=None, sparsity=10, save=True):
    if (id is None) or (id >= len(X) or id < 0):
        id = np.random.randint(0, len(X))
    sample = X[id]

    reconstructed = model.reconstruct(sample, sparsity)
    err = np.linalg.norm(sample - reconstructed)

    plt.plot(sample, label='original')
    plt.plot(reconstructed, label='reconstructed')
    plt.legend()
    plt.title(f'Example of reconstruction with {model_name} ({sparsity} atoms)')
    if save:
        plt.savefig(f'figures/example_{model_name}_{sparsity}_atoms.png')
    plt.show()

    print(f"Reconstruction error ({model_name}): {err:>.4f}")

def evaluate_error(model, model_name, X, sparsity):
    err = 0
    t = 0
    for i in range(len(X)):
        t0 = time.time()
        err += np.linalg.norm(X[i] - model.reconstruct(X[i], sparsity))
        dt = time.time() - t0
        t += dt
    err /= len(X)
    t /= len(X)
    print(f'Average reconstruction error for {model_name} model ({sparsity} atoms): {err:.4f}')
    print(f'Average reconstruction time for {model_name} model ({sparsity} atoms): {t:.4f}s')
    return err

def save_model(model, model_name):
    if 'TWI' in model_name:
        dict = model._D.copy()
    else:
        dict = model.D.copy()
    np.savez(f'models/{model_name}.npz', *dict)

def load_model(model_name):
    data = np.load(f'models/{model_name}.npz')
    dict = [data[key] for key in data]
    if not 'TWI' in model_name:
        dict = np.array(dict)
        new_model = kSVD(init_dict=dict)
    else:
        new_model = TWI_kSVD(init_dict=dict)
    return new_model