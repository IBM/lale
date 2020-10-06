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

import sklearn.neighbors

# class that follows scikit-learn conventions but lacks schemas,
# for the purpose of testing how to wrap an operator without schemas


class UnknownOp:
    def __init__(self, n_neighbors=5, algorithm="auto"):
        self._hyperparams = {"n_neighbors": n_neighbors, "algorithm": algorithm}

    def get_params(self, deep: bool = False):
        return self._hyperparams

    def fit(self, X, y):
        self._wrapped_model = sklearn.neighbors.KNeighborsClassifier(
            **self._hyperparams
        )

    def predict(self, X):
        return self._wrapped_model.predict(X)


class BadClassifier:
    def __init__(self, is_good=False):
        self._hyperparams = {"is_good": is_good}

    def get_params(self, deep: bool = False):
        return self._hyperparams

    def fit(self, X, y):
        assert False, "Bad fit method."

    def predict(self, X):
        assert False, "Bad predict method."
