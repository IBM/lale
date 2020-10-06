# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """Your Python environment does not have torch installed. You can install it with
                                pip install torch
                                or with
                                    pip install 'lale[full]'"""
    )


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
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

    def get_data(self):
        if self.y is None:
            return self.X
        else:
            return self.X, self.y
