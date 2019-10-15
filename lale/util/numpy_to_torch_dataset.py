from torch.utils.data import Dataset

class NumpyTorchDataset(Dataset):
    """Pytorch Dataset subclass that takes a numpy array and an optional label array."""

    def __init__(self, X, y=None):
        """X and y are the dataset and labels respectively.
        
        Parameters
        ----------
        X : numpy array
            Two dimensional dataset of input features.
        y : numpy array
            Labels
        """
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        label = None
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

    def get_data(self):
        if self.y is None:
            return self.X
        else:
            return self.X, self.y                