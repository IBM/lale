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

import numpy as np
import pandas as pd

import lale.datasets.openml


def fetch_adult_df(preprocess=False):
    """
    Fetch the `adult`_ dataset from OpenML and add `fairness_info`.
    It contains information about individuals from the 1994 U.S. census.
    The prediction task is a binary classification on whether the
    income of a person exceeds 50K a year. Without preprocessing,
    the dataset has 48,842 rows and 14 columns. There are two
    protected attributes, sex and race, and the disparate impact is
    0.23. The data includes both categorical and numeric columns, and
    has some missing values.

    .. _`adult`: https://www.openml.org/d/179

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      impute missing values;
      encode protected attributes in X as 0 or 1 to indicate privileged groups;
      encode labels in y as 0 or 1 to indicate favorable outcomes;
      and apply one-hot encoding to any remaining features in X that
      are categorical and not protecteded attributes.

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
        "adult", "classification", astype="pandas", preprocess=preprocess
    )
    orig_X = pd.concat([train_X, test_X]).sort_index()
    orig_y = pd.concat([train_y, test_y]).sort_index()
    if preprocess:
        sex = pd.Series(orig_X["sex_Male"] == 1, dtype=np.float64)
        race = pd.Series(orig_X["race_White"] == 1, dtype=np.float64)
        dropped_X = orig_X.drop(
            labels=[
                "race_Amer-Indian-Eskimo",
                "race_Asian-Pac-Islander",
                "race_Black",
                "race_Other",
                "race_White",
                "sex_Female",
                "sex_Male",
            ],
            axis=1,
        )
        encoded_X = dropped_X.assign(sex=sex, race=race)
        assert not encoded_X.isna().any().any()
        assert not orig_y.isna().any().any()
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "sex", "reference_group": [1]},
                {"feature": "race", "reference_group": [1]},
            ],
        }
        return encoded_X, orig_y, fairness_info
    else:
        fairness_info = {
            "favorable_labels": [">50K"],
            "protected_attributes": [
                {"feature": "race", "reference_group": ["White"]},
                {"feature": "sex", "reference_group": ["Male"]},
            ],
        }
        return orig_X, orig_y, fairness_info


def fetch_bank_df(preprocess=False):
    """
    Fetch the `bank-marketing`_ dataset from OpenML and add `fairness_info`.

    It contains information from marketing campaigns of a Portuguise
    bank.  The prediction task is a binary classification on whether
    the client will subscribe a term deposit. Without preprocessing,
    the dataset has 45,211 rows and 16 columns. There is one protected
    attribute, age, and the disparate impact of 0.84. The data
    includes both categorical and numeric columns, with no missing
    values.

    .. _`bank-marketing`: https://www.openml.org/d/1461

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      encode protected attributes in X as 0 or 1 to indicate privileged groups;
      encode labels in y as 0 or 1 to indicate favorable outcomes;
      and apply one-hot encoding to any remaining features in X that
      are categorical and not protecteded attributes.

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
        "bank-marketing", "classification", astype="pandas", preprocess=preprocess
    )
    orig_X = pd.concat([train_X, test_X]).sort_index()
    orig_y = pd.concat([train_y, test_y]).sort_index().astype(np.float64)
    column_map = {
        "v1": "age",
        "v2": "job",
        "v3": "marital",
        "v4": "education",
        "v5": "default",
        "v6": "balance",
        "v7": "housing",
        "v8": "loan",
        "v9": "contact",
        "v10": "day",
        "v11": "month",
        "v12": "duration",
        "v13": "campaign",
        "v14": "pdays",
        "v15": "previous",
        "v16": "poutcome",
    }
    if preprocess:

        def map_col(col):
            if col.find("_") == -1:
                return column_map[col]
            prefix, suffix = col.split("_")
            return column_map[prefix] + "_" + suffix

        orig_X.columns = [map_col(col) for col in orig_X.columns]
        age = pd.Series(orig_X["age"] >= 25, dtype=np.float64)
        encoded_X = orig_X.assign(age=age)
        encoded_y = pd.Series(orig_y == 0, dtype=np.float64, name=orig_y.name)
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "age", "reference_group": [1]},
            ],
        }
        return encoded_X, encoded_y, fairness_info
    else:
        orig_X.columns = [column_map[col] for col in orig_X.columns]
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "age", "reference_group": [[25, 1000]]},
            ],
        }
        return orig_X, orig_y, fairness_info


def fetch_compas_df(preprocess=False):
    """
    Fetch the `compas-two-years`_ dataset, also known as ProPublica recidivism, from OpenML and add `fairness_info`.

    It contains information about individuals with a binary
    classification for recidivism, indicating whether they were
    re-arrested within two years after the first arrest. Without
    preprocessing, the dataset has 5,287 rows and 13 columns.  There
    are two protected attributes, sex and race, and the disparate
    impact is 0.92.  The data includes only numeric columns, with no
    missing values.

    .. _`compas-two-years`: https://www.openml.org/d/42193

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True, compute column `race` from `race_caucasian`, and drop
      columns `race_african-american` and `race_caucasian`.

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
        "compas", "classification", astype="pandas", preprocess=False
    )
    orig_X = pd.concat([train_X, test_X]).sort_index().astype(np.float64)
    orig_y = pd.concat([train_y, test_y]).sort_index().astype(np.float64)
    if preprocess:
        race = pd.Series(orig_X["race_caucasian"] == 1, dtype=np.float64)
        dropped_X = orig_X.drop(
            labels=["race_african-american", "race_caucasian"], axis=1
        )
        encoded_X = dropped_X.assign(race=race)
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "sex", "reference_group": [1]},
                {"feature": "race", "reference_group": [1]},
            ],
        }
        return encoded_X, orig_y, fairness_info
    else:
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "sex", "reference_group": [1]},
                {"feature": "race_caucasian", "reference_group": [1]},
            ],
        }
        return orig_X, orig_y, fairness_info


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
      are categorical and not protecteded attributes.

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


def fetch_ricci_df(preprocess=False):
    """
    Fetch the `ricci_vs_destefano`_ dataset from OpenML and add `fairness_info`.

    It contains test scores for 2003 New Haven Fire Department
    promotion exams with a binary classification into promotion or no
    promotion.  Without preprocessing, the dataset has 118 rows and 5
    columns.  There is one protected attribute, race, and the
    disparate impact is 0.50.  The data includes both categorical and
    numeric columns, with no missing values.

    .. _`ricci_vs_destefano`: https://www.openml.org/d/42665

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      encode protected attributes in X as 0 or 1 to indicate privileged groups;
      encode labels in y as 0 or 1 to indicate favorable outcomes;
      and apply one-hot encoding to any remaining features in X that
      are categorical and not protecteded attributes.

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
        "ricci", "classification", astype="pandas", preprocess=preprocess
    )
    orig_X = pd.concat([train_X, test_X]).sort_index()
    orig_y = pd.concat([train_y, test_y]).sort_index()
    if preprocess:
        race = pd.Series(orig_X["race_W"] == 1, dtype=np.float64)
        dropped_X = orig_X.drop(labels=["race_B", "race_H", "race_W"], axis=1)
        encoded_X = dropped_X.assign(race=race)
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [{"feature": "race", "reference_group": [1]}],
        }
        return encoded_X, orig_y, fairness_info
    else:
        fairness_info = {
            "favorable_labels": ["Promotion"],
            "protected_attributes": [{"feature": "race", "reference_group": ["W"]}],
        }
        return orig_X, orig_y, fairness_info
