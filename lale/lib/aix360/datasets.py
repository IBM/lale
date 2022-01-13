# Copyright 2020,2021 IBM Corporation
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
# limitations under the Lic
import logging

import numpy as np
import pandas as pd

import lale.datasets
import lale.datasets.openml
import lale.lib.aif360.util

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def fetch_creditg_df(preprocess=False):
    """
    Fetch the `credit-g`_ dataset from OpenML and add `fairness_info`.

    It contains information about individuals with a binary
    classification into good or bad credit risks. Without
    preprocessing, the dataset has 1,000 rows and 20 columns. There
    are two protected attributs, personal_status/sex and age, and the
    disparate impact is 0.75.  The data includes both categorical and
    numeric columns, with no missing values.

    .. _`credit-g`: https://www.openml.org/d/31

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      encode protected attributes in X as 0 or 1 to indicate privileged groups;
      encode labels in y as 0 or 1 to indicate favorable outcomes;
      and apply one-hot encoding to any remaining features in X that
      are categorical and not protected attributes.

    Returns
    -------
    result : tuple

      - item 0: pandas Dataframe

          Features X, including both protected and non-protected attributes.

      - item 1: pandas Series

          Labels y.

      - item 3: fairness_info

          JSON meta-data following the format understood by fairness metrics
          and mitigation operators in `lale.lib.aif360`.
    """
    (train_X, train_y), (test_X, test_y) = lale.datasets.openml.fetch(
        "credit-g", "classification", astype="pandas", preprocess=preprocess
    )
    orig_X = pd.concat([train_X, test_X]).sort_index()
    orig_y = pd.concat([train_y, test_y]).sort_index()
    if preprocess:
        sex = pd.Series(
            (orig_X["personal_status_male div/sep"] == 1)
            | (orig_X["personal_status_male mar/wid"] == 1)
            | (orig_X["personal_status_male single"] == 1),
            dtype=np.float64,
        )
        age = pd.Series(orig_X["age"] > 25, dtype=np.float64)
        dropped_X = orig_X.drop(
            labels=[
                "personal_status_female div/dep/mar",
                "personal_status_male div/sep",
                "personal_status_male mar/wid",
                "personal_status_male single",
            ],
            axis=1,
        )
        encoded_X = dropped_X.assign(sex=sex, age=age)
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "sex", "reference_group": [1]},
                {"feature": "age", "reference_group": [1]},
            ],
        }
        return encoded_X, orig_y, fairness_info
    else:
        fairness_info = {
            "favorable_labels": ["good"],
            "protected_attributes": [
                {
                    "feature": "personal_status",
                    "reference_group": [
                        "male div/sep",
                        "male mar/wid",
                        "male single",
                    ],
                },
                {"feature": "age", "reference_group": [[26, 1000]]},
            ],
        }
        return orig_X, orig_y, fairness_info
