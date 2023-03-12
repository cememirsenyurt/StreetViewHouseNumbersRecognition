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
    return mat_contents['X'], mat_contents['y']
    

if __name__ == "__main__":
    X, y = loadData("train")
