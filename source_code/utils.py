import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

np.random.seed(0)

def load_DIGITS_dataset(dataset_type):
    with open(f'datasets/digits/{dataset_type}_D.pickle', 'rb') as f:
        dataset = pickle.load(f)
    return dataset['X'], dataset['Y'], dataset['matrix_X'], dataset['matrix_Y'], dataset['labels']

def load_BME_dataset(dataset_type):
    with open(f'datasets/BME/{dataset_type}_D.pickle', 'rb') as f:
        dataset = pickle.load(f)
    return dataset['X'], dataset['labels']
    
def get_dataset(dataset_name, dataset_type):
    if dataset_name == 'BME':
        return load_BME_dataset(dataset_type)

    if dataset_name == 'DIGITS':
        return load_DIGITS_dataset(dataset_type)

    raise ValueError(f'Unknown dataset name: {dataset_name}')

def rotation(a, b, c):
    len = a.shape[0]
    theta = np.arccos((b.T @ c)/(np.linalg.norm(b) * np.linalg.norm(c)))[0, 0]
    u = b / np.linalg.norm(b)
    v = (c - (u.T @ c)*u)/np.linalg.norm((c - (u.T @ c)*u))
    
    R_theta = np.hstack([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).reshape(2, 2)
    R = np.eye(len) - u @ u.T - v @ v.T + np.hstack([u, v]) @ R_theta @ np.hstack([u, v]).T
    ar = R @ a
    return ar

def init_dictionnary(X, n_classes):
    signal_size = X.shape[0]
    dictionary_list = np.random.random((signal_size, 10*n_classes))
    return dictionary_list

def plot_example(x, y, model, sparsity):
    reconstruction, y_pred = model.reconstruct(x, sparsity=sparsity)

    plt.figure() 
    plt.plot(x, label='Original')
    plt.plot(reconstruction, label='Reconstruction')
    plt.legend()
    plt.title(f'Original signal (class {y}) vs. reconstruction (class {y_pred})')
    plt.show()

def get_errors_1d(model, X, y, sparsity=10):
    y_pred = []
    l2_error = 0
    for i in range(X.shape[1]):
        x = X[:, i]
        reconstruction, y_pred_i = model.reconstruct(x, sparsity=sparsity)
        y_pred.append(y_pred_i)
        l2_error += np.linalg.norm(x - reconstruction)
    y_pred = np.array(y_pred)
    accuracy = np.mean(y_pred == y)
    classif_error = 1 - accuracy
    l2_error /= X.shape[1]
    return classif_error, l2_error

def get_errors_1d_array(model_1D, X_test_array, X_test, test_labels, sparsity=10):
    accuracy = 0
    l2_error = 0
    for i in range(X_test_array.shape[1]):
        x = X_test[i]
        true_len = len(x)
        true_label = test_labels[i]

        x_array = X_test_array[:, i]

        reconstruction_x, pred_label = model_1D.reconstruct(x_array, sparsity=sparsity)
        accuracy += int(pred_label == true_label)
        l2_error += np.linalg.norm(x - reconstruction_x[:true_len])

    accuracy /= X_test_array.shape[1]
    error_rate = 1 - accuracy
    l2_error /= X_test_array.shape[1]
    return error_rate, l2_error

def get_errors_2d_array(model_2D, X_test_array, Y_test_array, X_test, Y_test, test_labels, sparsity=10):
    accuracy = 0
    l2_error = 0
    for i in range(X_test_array.shape[1]):
        x = X_test[i]
        y = Y_test[i]
        true_len = len(x)
        true_label = test_labels[i]

        x_array = X_test_array[:, i]
        y_array = Y_test_array[:, i]

        reconstruction_x, reconstruction_y, pred_label = model_2D.reconstruct(x_array, y_array, sparsity=sparsity)
        accuracy += int(pred_label == true_label)
        l2_error += np.sqrt(np.linalg.norm(x - reconstruction_x[:true_len])**2 + np.linalg.norm(y - reconstruction_y[:true_len])**2)

    accuracy /= X_test_array.shape[1]
    error_rate = 1 - accuracy
    l2_error /= X_test_array.shape[1]
    return error_rate, l2_error