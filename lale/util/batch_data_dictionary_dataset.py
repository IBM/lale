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
  
class BatchDataDictDataset(Dataset):
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
        self.X = X
        self.y = y
        self.num_batches = len(X)
        first_batch = X[0]
        if isinstance(first_batch, tuple):
            X_0 = first_batch[0]
        self.batch_size = X_0.shape[0]
        batch_count = 0
        self.small_batch_idx = None
        self.small_batch_size = None
        for batch_idx in X.keys():
            batch_data = X[batch_idx]
            if isinstance(batch_data, tuple):
                X_t = batch_data[0]
            batch_count +=1
            if X_t.shape[0] <self.batch_size:
                self.small_batch_idx = batch_idx
                self.small_batch_size = X_t.shape[0]
        if self.small_batch_size is None:
            self.small_batch_size = self.batch_size
            self.small_batch_idx = self.num_batches-1
        # #Swap the small batch and last batch to allow handling of variable length sequences in general
        # temp_batch = X[self.small_batch_idx]
        # X[self.small_batch_idx] = X[self.num_batches-1]
        # X[self.num_batches-1] = temp_batch
        # self.small_batch_idx = -1


    def __len__(self):
        return self.batch_size*(self.num_batches-1)+self.small_batch_size

    def __getitem__(self, idx):
        batch_idx = idx//self.batch_size
        if batch_idx == self.small_batch_idx:
            id_within_batch = idx%self.batch_size
            if id_within_batch >= self.small_batch_size:
                batch_idx +=1
                id_within_batch = id_within_batch - self.small_batch_size
        elif batch_idx > self.small_batch_idx:
            id_within_batch = idx%self.batch_size
            if id_within_batch >= self.small_batch_size:
                batch_idx +=1
                id_within_batch = id_within_batch - self.small_batch_size
            else:
                id_within_batch = id_within_batch + self.batch_size - self.small_batch_size 
        else:
            id_within_batch = idx%self.batch_size
        batch_data = self.X[batch_idx]
        if isinstance(batch_data, tuple):
            return batch_data[0][id_within_batch], batch_data[1][id_within_batch]
        else:
            return batch_data[id_within_batch]

    def get_data(self):
        return self.X
