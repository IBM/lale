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

from torch.utils.data import Dataset


class BatchDataDict(Dataset):
    """Pytorch Dataset subclass that takes a dictionary of format {'<batch_idx>': <batch_data>}."""

    def __init__(self, X, y=None):
        """X is the dictionary dataset and y is ignored.

        Parameters
        ----------
        X : dict
         Dictionary of format {'<batch_idx>': <batch_data>}
        y : None
            Ignored.
        """
        self.data_dict = X

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        # This returns the batch at idx instead of a single element.
        return self.data_dict[idx]
