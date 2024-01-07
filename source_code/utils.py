import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

from source_code.models_ksvd import kSVD, kSVD_2D
from source_code.models_twi import TWI_kSVD, TWI_kSVD_2D

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
    _a = a.copy()
    _b = b.copy()
    _c = c.copy()

    len_ = _a.shape[0]
    angle = (np.dot(_b.T, _c))/(np.linalg.norm(_b) * np.linalg.norm(_c))
    angle = angle[0, 0]
    if angle > 1:
        angle = 1
    theta = np.arccos(angle)

    u = _b / np.linalg.norm(_b)
    v = _c - np.dot(u.T, _c)*u
    v /= np.linalg.norm(v)
    
    R_theta = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R = np.eye(len_) - u @ u.T - v @ v.T + np.hstack([u, v]) @ R_theta @ np.hstack([u, v]).T
    ar = (R @ _a).reshape(-1, 1)

    return ar

def init_dictionnary(X, n_atoms):
    dictionary = np.zeros((len(X), n_atoms))
    indices = np.random.choice(np.arange(X.shape[1]), n_atoms)
    for i in range(n_atoms):
        sample = X[:, indices[i]]
        start = np.random.randint(0, len(sample)//4)
        end = start + 3*len(sample)//4
        new_atom = sample[start:end]
        dictionary[start:end, i] = new_atom
    return dictionary/np.linalg.norm(dictionary, axis=0)

def init_list_dictionnary(X, n_atoms):
    indices = np.random.choice(np.arange(len(X)), n_atoms)
    dictionary_list = []
    for i in range(n_atoms):
        sample = X[indices[i]].copy()
        start = np.random.randint(0, len(sample)//4)
        end = start + 3*len(sample)//4
        new_atom = sample[start:end]
        new_atom /= np.linalg.norm(new_atom)
        dictionary_list.append(new_atom)
    return dictionary_list

def get_errors_1d_array(model, X, y, sparsity, only_l2=False):
    y_pred, my_y_pred = [], []
    l2_error = 0
    for i in range(X.shape[1]):
        x = X[:, i]
        reconstruction, y_pred_i, my_y_pred_i = model.reconstruct(x, sparsity=sparsity, only_l2=only_l2)
        y_pred.append(y_pred_i)
        my_y_pred.append(my_y_pred_i)
        l2_error += np.linalg.norm(x - reconstruction)/np.linalg.norm(x)
    y_pred = np.array(y_pred)
    my_y_pred = np.array(my_y_pred)
    accuracy = np.mean(y_pred == y)
    my_accuracy = np.mean(my_y_pred == y)
    classif_error = 1 - accuracy
    my_classif_error = 1 - my_accuracy
    l2_error /= X.shape[1]
    return classif_error, my_classif_error, l2_error

def get_errors_1d_list(model, _list, y, sparsity, only_l2=False):
    y_pred, my_y_pred = [], []
    l2_error = 0
    for i in range(len(_list)):
        x = _list[i]
        reconstruction, y_pred_i, my_y_pred_i = model.reconstruct(x, sparsity=sparsity, only_l2=only_l2)
        y_pred.append(y_pred_i)
        my_y_pred.append(my_y_pred_i)
        error = np.linalg.norm(x - reconstruction)/np.linalg.norm(x)
        l2_error += error
    y_pred = np.array(y_pred)
    my_y_pred = np.array(my_y_pred)
    accuracy = np.mean(y_pred == y)
    my_accuracy = np.mean(my_y_pred == y)

    classif_error = 1 - accuracy
    my_classif_error = 1 - my_accuracy
    l2_error /= len(_list)
    return classif_error, my_classif_error, l2_error

def get_errors_2d_array(model_2D, X_test_array, Y_test_array, X_test, Y_test, test_labels, sparsity, only_l2=False):
    y_pred, my_y_pred = [], []
    l2_error = 0
    for i in range(X_test_array.shape[1]):
        x = X_test[i]
        y = Y_test[i]
        true_len = len(x)
        true_label = test_labels[i]

        x_array = X_test_array[:, i]
        y_array = Y_test_array[:, i]

        reconstruction_x, reconstruction_y, pred_label, my_pred_label = model_2D.reconstruct(x_array, y_array, sparsity=sparsity, only_l2=only_l2)
        y_pred.append(pred_label)
        my_y_pred.append(my_pred_label)

        error = np.linalg.norm(x - reconstruction_x[:true_len])/np.linalg.norm(x)
        error += np.linalg.norm(y - reconstruction_y[:true_len])/np.linalg.norm(y)
        l2_error += error

    accuracy = np.mean(np.array(y_pred) == test_labels)
    my_accuracy = np.mean(np.array(my_y_pred) == test_labels)
    classif_error = 1 - accuracy
    my_classif_error = 1 - my_accuracy
    l2_error /= X_test_array.shape[1]
    return classif_error, my_classif_error, l2_error

def get_errors_2d_list(model_2D, X, Y, labels, sparsity, only_l2=False):
    y_pred, my_y_pred = [], []
    l2_error = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        true_label = labels[i]

        reconstruction_x, reconstruction_y, pred_label, my_pred_label = model_2D.reconstruct(x, y, sparsity=sparsity, only_l2=only_l2)
        y_pred.append(pred_label)
        my_y_pred.append(my_pred_label)

        error = np.linalg.norm(x - reconstruction_x)/np.linalg.norm(x)
        error += np.linalg.norm(y - reconstruction_y)/np.linalg.norm(y)
        l2_error += error

    accuracy = np.mean(np.array(y_pred) == labels)
    my_accuracy = np.mean(np.array(my_y_pred) == labels)
    classif_error = 1 - accuracy
    my_classif_error = 1 - my_accuracy
    l2_error /= len(X)
    return classif_error, my_classif_error, l2_error

def save_model(model, model_name):
    D_list = model.D_list.copy()
    with open(f'models/{model_name}.pkl', 'wb') as file:
        pickle.dump(D_list, file)

def load_model(model_name):
    with open(f'models/{model_name}.pkl', 'rb') as file:
        D_list = pickle.load(file)
    n_classes = len(D_list)
    n_atoms = len(D_list[0])
    if 'TWI' in model_name:
        new_model = TWI_kSVD(n_classes=n_classes, init_D_list=D_list)
    else:
        new_model = kSVD(n_classes=n_classes, init_D_list=D_list)
    return new_model

def save_2d_model(model, model_name):
    D_list_X = model.D_list_X.copy()
    D_list_Y = model.D_list_Y.copy()

    with open(f'models/{model_name}_X.pkl', 'wb') as file:
        pickle.dump(D_list_X, file)
    with open(f'models/{model_name}_Y.pkl', 'wb') as file:
        pickle.dump(D_list_Y, file)

def load_2d_model(model_name):
    with open(f'models/{model_name}_X.pkl', 'rb') as file:
        D_list_X = pickle.load(file)
    with open(f'models/{model_name}_Y.pkl', 'rb') as file:
        D_list_Y = pickle.load(file)
    D_list = [D_list_X, D_list_Y]
    n_classes = len(D_list_X)
    n_atoms = len(D_list_X[0])
    if 'TWI' in model_name:
        new_model = TWI_kSVD_2D(n_classes=n_classes, init_D_list=D_list)
    else:
        new_model = kSVD_2D(n_classes=n_classes, init_D_list=D_list)
    return new_model