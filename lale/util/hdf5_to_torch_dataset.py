from torch.utils.data import Dataset
import h5py

class HDF5TorchDataset(Dataset):
    """Pytorch Dataset subclass that takes a hdf5 file pointer."""

    def __init__(self, file_path):
        """.
        
        Parameters
        ----------
        file : file is an object of class h5py.File
        """
        self.file_path = file_path
        h5_file = h5py.File(file_path)
        self.length = h5_file['X'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path) as h5_file:
            X = h5_file['X']
            try:
                y = h5_file['y']
            except KeyError:
                y = None
            if y is None:
                element = X[idx]
            else:
                element = X[idx], y[idx]
        return element

    def get_data(self):
        with h5py.File(self.file_path) as h5_file:
            X = h5_file['X'][:]
            try:
                y = h5_file['y'][:]
            except KeyError:
                y = None
            if y is None:
                return X
            else:
                return X, y
