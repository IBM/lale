# Copyright 2021 IBM Corporation
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

import pandas as pd

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """Your Python environment does not have torch installed. You can install it with
                                pip install torch
                                or with
                                    pip install 'lale[full]'"""
    )


class PandasTorchDataset(Dataset):
    """Pytorch Dataset subclass that takes a pandas DataFrame and an optional label pandas Series."""

    def __init__(self, X, y=None):
        """X and y are the dataset and labels respectively.

        Parameters
        ----------
        X : pandas DataFrame
            Two dimensional dataset of input features.
        y : pandas Series
            Labels
        """
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X.iloc[idx], self.y.iloc[idx]
        else:
            return self.X.iloc[idx]

    def get_data(self):
        if self.y is None:
            return self.X
        else:
            return self.X, self.y


def pandas_collate_fn(batch):
    return_X = None
    return_y = None

    for item in batch:
        if isinstance(item, tuple):
            if return_X is None:
                return_X = [item[0].to_dict()]
            else:
                return_X.append(item[0].to_dict())
            if return_y is None:
                return_y = [item[1]]
            else:
                return_y.append(item[1])
        else:
            if return_X is None:
                return_X = [item.to_dict()]
            else:
                return_X.append(item.to_dict())
    if return_y is not None:
        return (pd.DataFrame(return_X), pd.Series(return_y))
    else:
        return pd.DataFrame(return_X)
