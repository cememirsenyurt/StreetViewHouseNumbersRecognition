import scipy.io as sio
import numpy as np
import os

def loadData(dataset: str) -> "tuple[np.ndarray, np.ndarray]":
    '''
    Loads and returns labelled SVHN data from one of three datasets.
    Must be called in a directory which contains a dataset/ folder
    containing the .mat files.
    '''
    mat_names = {
        "train": "train_32x32.mat",
        "test": "test_32x32.mat",
        "extra": "extra_32x32.mat"
    }
    if dataset not in mat_names.keys():
        raise Exception("Invalid dataset")
    mat_name = mat_names[dataset]
    mat_path = os.path.join(os.path.curdir, "datasets", mat_name)
    mat_contents = sio.loadmat(mat_path)
    X, y = mat_contents['X'], mat_contents['y']
    #convert X to [0,1]
    X = X/255
    #swap index axis to first axis
    X = np.swapaxes(X,3,2)
    X = np.swapaxes(X,2,1)
    X = np.swapaxes(X,1,0)
    #turn 10s into 0s for simplicity
    y = np.where(y == 10, 0, y)
    return X, y

def loadDataFromPath(relative_path: str) -> "tuple[np.ndarray, np.ndarray]":
    '''
    Loads and returns labelled SVHN data from the dataset located at p,
    a path relative to the current directory.
    '''
    mat_path = os.path.join(os.path.curdir, relative_path)
    mat_contents = sio.loadmat(mat_path)
    X, y = mat_contents['X'], mat_contents['y']
    #convert X to [0,1]
    X = X/255
    #swap index axis to first axis
    X = np.swapaxes(X,3,2)
    X = np.swapaxes(X,2,1)
    X = np.swapaxes(X,1,0)
    #turn 10s into 0s for simplicity
    y = np.where(y == 10, 0, y)
    return X, y
    

if __name__ == "__main__":
    X, y = loadData("train")
