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

import sklearn.metrics
import sklearn.model_selection
from sklearn.utils.metaestimators import _safe_split

from lale.datasets.data_schemas import add_folds_for_monoid


class FoldsForMonoid:
    def __init__(self, estimator, X, y, cv):
        self.train_Xs = []
        self.train_ys = []
        self.test_Xs = []
        self.test_ys = []
        for split_id, (train, test) in enumerate(cv.split(X, y)):
            train_X, train_y = _safe_split(estimator, X, y, train)
            train_X = add_folds_for_monoid(train_X, (split_id, self))
            train_y = add_folds_for_monoid(train_y, (split_id, self))
            self.train_Xs.append(train_X)
            self.train_ys.append(train_y)
            test_X, test_y = _safe_split(estimator, X, y, test, train)
            test_X = add_folds_for_monoid(test_X, (split_id, self))
            test_y = add_folds_for_monoid(test_y, (split_id, self))
            self.test_Xs.append(test_X)
            self.test_ys.append(test_y)
        self._split2op2lifted = [{} for _ in range(cv.get_n_splits())]

    def get_lifted(self, split_id, op_id):
        return self._split2op2lifted[split_id][op_id]

    def has_lifted(self, split_id, op_id):
        return op_id in self._split2op2lifted[split_id]

    def set_lifted(self, split_id, op_id, lifted):
        self._split2op2lifted[split_id][op_id] = lifted

    def get_n_splits(self):
        return len(self.train_Xs)


def cross_val_score_for_monoid(
    estimator,
    X,
    y=None,
    scoring=sklearn.metrics.accuracy_score,
    cv=5,
    return_estimators=False,
):
    if isinstance(cv, int):
        cv = sklearn.model_selection.StratifiedKFold(cv)
    folds = FoldsForMonoid(estimator, X, y, cv)
    estimators = []
    cv_results = []
    for i in range(folds.get_n_splits()):
        trained_estimator = estimator.fit(folds.train_Xs[i], folds.train_ys[i])
        if return_estimators:
            estimators.append(trained_estimator)
        predicted_values = trained_estimator.predict(folds.test_Xs[i])
        scoring_result = scoring(folds.test_ys[i], predicted_values)
        cv_results.append(scoring_result)
    if return_estimators:
        return estimators, cv_results
    return cv_results
