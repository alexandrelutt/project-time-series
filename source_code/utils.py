import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff

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