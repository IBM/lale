# Copyright 2019-2022 IBM Corporation
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

from typing import Tuple

import pandas as pd
import sklearn.model_selection

from lale.datasets.data_schemas import add_schema_adjusting_n_rows

from .util import _validate_fairness_info


def _column_for_stratification(
    X, y, favorable_labels, protected_attributes, unfavorable_labels
):
    from lale.lib.aif360 import ProtectedAttributesEncoder

    prot_attr_enc = ProtectedAttributesEncoder(
        favorable_labels=favorable_labels,
        protected_attributes=protected_attributes,
        unfavorable_labels=unfavorable_labels,
        remainder="drop",
        return_X_y=True,
    )
    encoded_X, encoded_y = prot_attr_enc.transform(X, y)
    df = pd.concat([encoded_X, encoded_y], axis=1)

    def label_for_stratification(row):
        return "".join(["T" if v == 1 else "F" if v == 0 else "N" for v in row])

    result = df.apply(label_for_stratification, axis=1)
    result.name = "stratify"
    return result


def fair_stratified_train_test_split(
    X,
    y,
    *arrays,
    favorable_labels,
    protected_attributes,
    unfavorable_labels=None,
    test_size=0.25,
    random_state=None,
) -> Tuple:
    """
    Splits X and y into random train and test subsets stratified by
    labels and protected attributes.

    Behaves similar to the `train_test_split`_ function from scikit-learn.

    .. _`train_test_split`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    Parameters
    ----------
    X : array

      Features including protected attributes as numpy ndarray or pandas dataframe.

    y : array

      Labels as numpy ndarray or pandas series.

    *arrays : array

      Sequence of additional arrays with same length as X and y.

    favorable_labels : array

      Label values which are considered favorable (i.e. "positive").

    protected_attributes : array

      Features for which fairness is desired.

    unfavorable_labels : array or None, default None

      Label values which are considered unfavorable (i.e. "negative").

    test_size : float or int, default=0.25

      If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
      If int, represents the absolute number of test samples.

    random_state : int, RandomState instance or None, default=None

      Controls the shuffling applied to the data before applying the split.
      Pass an integer for reproducible output across multiple function calls.

      - None

          RandomState used by numpy.random

      - numpy.random.RandomState

          Use the provided random state, only affecting other users of that same random state instance.

      - integer

          Explicit seed.

    Returns
    -------
    result : tuple

      - item 0: train_X

      - item 1: test_X

      - item 2: train_y

      - item 3: test_y

      - item 4+: Each argument in `*arrays`, if any, yields two items in the result, for the two splits of that array.
    """
    _validate_fairness_info(
        favorable_labels, protected_attributes, unfavorable_labels, True
    )
    stratify = _column_for_stratification(
        X, y, favorable_labels, protected_attributes, unfavorable_labels
    )
    (
        train_X,
        test_X,
        train_y,
        test_y,
        *arrays_splits,
    ) = sklearn.model_selection.train_test_split(
        X, y, *arrays, test_size=test_size, random_state=random_state, stratify=stratify
    )
    if hasattr(X, "json_schema"):
        train_X = add_schema_adjusting_n_rows(train_X, X.json_schema)
        test_X = add_schema_adjusting_n_rows(test_X, X.json_schema)
    if hasattr(y, "json_schema"):
        train_y = add_schema_adjusting_n_rows(train_y, y.json_schema)
        test_y = add_schema_adjusting_n_rows(test_y, y.json_schema)
    return (train_X, test_X, train_y, test_y, *arrays_splits)


class FairStratifiedKFold:
    """
    Stratified k-folds cross-validator by labels and protected attributes.

    Behaves similar to the `StratifiedKFold`_ class from scikit-learn.
    This cross-validation object can be passed to the `cv` argument of
    the `auto_configure`_ method.

    .. _`StratifiedKFold`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    .. _`auto_configure`: https://lale.readthedocs.io/en/latest/modules/lale.operators.html#lale.operators.PlannedOperator.auto_configure
    """

    def __init__(
        self,
        favorable_labels,
        protected_attributes,
        unfavorable_labels=None,
        n_splits=5,
        shuffle=False,
        random_state=None,
    ):
        """
        Parameters
        ----------
        favorable_labels : array

          Label values which are considered favorable (i.e. "positive").

        protected_attributes : array

          Features for which fairness is desired.

        unfavorable_labels : array or None, default None

          Label values which are considered unfavorable (i.e. "negative").

        n_splits : integer, optional, default 5

          Number of folds. Must be at least 2.

        shuffle : boolean, optional, default False

          Whether to shuffle each class's samples before splitting into batches.

        random_state : union type, not for optimizer, default None

          When shuffle is True, random_state affects the ordering of the indices.

          - None

              RandomState used by np.random

          - numpy.random.RandomState

              Use the provided random state, only affecting other users of that same random state instance.

          - integer

              Explicit seed.
        """
        _validate_fairness_info(
            favorable_labels, protected_attributes, unfavorable_labels, True
        )
        self._fairness_info = {
            "favorable_labels": favorable_labels,
            "protected_attributes": protected_attributes,
            "unfavorable_labels": unfavorable_labels,
        }
        self._stratified_k_fold = sklearn.model_selection.StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        The number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : Any

            Always ignored, exists for compatibility.

        y : Any

            Always ignored, exists for compatibility.

        groups : Any

            Always ignored, exists for compatibility.

        Returns
        -------
        integer
            The number of splits.
        """
        return self._stratified_k_fold.get_n_splits(X, y, groups)

    def split(self, X, y, groups=None):
        """
        Generate indices to split data into training and test set.

        X : array **of** items : array **of** items : Any

            Training data, including columns with the protected attributes.

        y : union type

            Target class labels; the array is over samples.

            - array **of** items : float

            - array **of** items : string

        groups : Any

            Always ignored, exists for compatibility.

        Yields
        ------
        result : tuple

            - train

                The training set indices for that split.

            - test

                The testing set indices for that split.
        """
        stratify = _column_for_stratification(X, y, **self._fairness_info)
        result = self._stratified_k_fold.split(X, stratify, groups)
        return result
