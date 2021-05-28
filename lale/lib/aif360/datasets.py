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

import os
from enum import Enum

import aif360.datasets
import numpy as np
import pandas as pd

import lale.datasets
import lale.datasets.openml
import lale.lib.aif360.util


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


def fetch_speeddating_df(preprocess=False):
    """
    Fetch the `SpeedDating`_ dataset from OpenML and add `fairness_info`.

    It contains data gathered from participants in experimental speed dating events
    from 2002-2004 with a binary classification into match or no
    match.  Without preprocessing, the dataset has 8378 rows and 122
    columns.  There are two protected attributes, whether the other candidate has the same
    race and importance of having the same race, and the disparate impact
    is 0.85.  The data includes both categorical and
    numeric columns, with some missing values.

    .. _`SpeedDating`: https://www.openml.org/d/40536

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
        "SpeedDating", "classification", astype="pandas", preprocess=preprocess
    )
    orig_X = pd.concat([train_X, test_X]).sort_index()
    orig_y = pd.concat([train_y, test_y]).sort_index()
    if preprocess:
        importance_same_race = pd.Series(
            orig_X["importance_same_race"] >= 9, dtype=np.float64
        )
        samerace = pd.Series(orig_X["samerace_1"] == 1, dtype=np.float64)
        dropped_X = orig_X.drop(labels=["samerace_0", "samerace_1"], axis=1)
        encoded_X = dropped_X.assign(
            samerace=samerace, importance_same_race=importance_same_race
        )
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "samerace", "reference_group": [1]},
                {"feature": "importance_same_race", "reference_group": [1]},
            ],
        }
        return encoded_X, orig_y, fairness_info
    else:
        fairness_info = {
            "favorable_labels": ["1"],
            "protected_attributes": [
                {"feature": "samerace", "reference_group": ["1"]},
                {"feature": "importance_same_race", "reference_group": [[9, 1000]]},
            ],
        }
        return orig_X, orig_y, fairness_info


def fetch_boston_housing_df(preprocess=False):
    """
    Fetch the `Boston housing`_ dataset from sklearn and add `fairness info`.

    It contains data about housing values in the suburbs of Boston with various
    features that can be used to perform regression. Without preprocessing,
    the dataset has 506 rows and 14 columns. There is one protected attribute,
    1000(Bk - 0.63)^2 where Bk is the proportion of Blacks by town, and the disparate
    impact is 0.5. The data includes only numeric columns, with no missing values.

    .. _`Boston housing`: https://scikit-learn.org/0.20/datasets/index.html#boston-house-prices-dataset

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      encode protected attribute in X as 0 or 1 to indicate privileged groups.

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
    (train_X, train_y), (test_X, test_y) = lale.datasets.boston_housing_df(
        test_size=0.33
    )
    orig_X = pd.concat([train_X, test_X]).sort_index()
    orig_y = pd.concat([train_y, test_y]).sort_index()
    black_median = np.median(train_X["B"])
    label_median = np.median(train_y)

    if preprocess:
        # 1000(Bk - 0.63)^2 where Bk is the proportion of Blacks by town
        B = pd.Series(orig_X["B"] > black_median, dtype=np.float64)
        encoded_X = orig_X.assign(B=B)
        fairness_info = {
            "favorable_labels": [[-10000.0, label_median]],
            "protected_attributes": [
                {"feature": "B", "reference_group": [0]},
            ],
        }
        return encoded_X, orig_y, fairness_info
    else:
        fairness_info = {
            "favorable_labels": [[-10000.0, label_median]],
            "protected_attributes": [
                # 1000(Bk - 0.63)^2 where Bk is the proportion of Blacks by town
                {"feature": "B", "reference_group": [[0.0, black_median]]},
            ],
        }
        return orig_X, orig_y, fairness_info


# MEPS HELPERS
class FiscalYear(Enum):
    FY2015 = 15
    FY2016 = 16


class Panel(Enum):
    PANEL19 = 19
    PANEL20 = 20
    PANEL21 = 21


def race(row):
    if (row["HISPANX"] == 2) and (
        row["RACEV2X"] == 1
    ):  # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
        return "White"
    return "Non-White"


def get_utilization_columns(fiscal_year):
    return [
        f"OBTOTV{fiscal_year.value}",
        f"OPTOTV{fiscal_year.value}",
        f"ERTOT{fiscal_year.value}",
        f"IPNGTD{fiscal_year.value}",
        f"HHTOTD{fiscal_year.value}",
    ]


def get_total_utilization(row, fiscal_year):
    cols = get_utilization_columns(fiscal_year)
    return sum(list(map(lambda x: row[x], cols)))


def should_drop_column(x, fiscal_year):
    utilization_cols = set(get_utilization_columns(fiscal_year))
    return x in utilization_cols


def fetch_meps_raw_df(panel, fiscal_year):
    filename = ""
    if fiscal_year == FiscalYear.FY2015:
        filename = "h181.csv"
    elif fiscal_year == FiscalYear.FY2016:
        filename = "h192.csv"
    else:
        raise ValueError(f"Unexpected FiscalYear received: {fiscal_year}")
    filepath = os.path.join(
        os.path.dirname(os.path.abspath(aif360.__file__)),
        "data",
        "raw",
        "meps",
        filename,
    )

    try:
        df = pd.read_csv(filepath, sep=",", na_values=[])
    except IOError as err:
        print("IOError: {}".format(err))
        print("To use this class, please follow the instructions in found here:")
        print(
            "\n\t{}\n".format(
                "https://github.com/Trusted-AI/AIF360/tree/master/aif360/data/raw/meps"
            )
        )
        print(
            f"\n to download and convert the data and place the final {filename} file, as-is, in the folder:"
        )
        print(
            "\n\t{}\n".format(
                os.path.abspath(
                    os.path.join(
                        os.path.abspath(__file__), "..", "..", "data", "raw", "meps"
                    )
                )
            )
        )
        import sys

        sys.exit(1)

    df["RACEV2X"] = df.apply(lambda row: race(row), axis=1)
    df = df.rename(columns={"RACEV2X": "RACE"})
    df = df[df["PANEL"] == panel.value]

    df["TOTEXP15"] = df.apply(
        lambda row: get_total_utilization(row, fiscal_year), axis=1
    )
    lessE = df["TOTEXP15"] < 10.0
    df.loc[lessE, "TOTEXP15"] = 0.0
    moreE = df["TOTEXP15"] >= 10.0
    df.loc[moreE, "TOTEXP15"] = 1.0

    df = df.rename(columns={"TOTEXP15": "UTILIZATION"})
    columns_to_drop = set(
        filter(lambda x: should_drop_column(x, fiscal_year), df.columns.tolist())
    )
    df = df[sorted(set(df.columns.tolist()) - columns_to_drop, key=df.columns.get_loc)]
    X = pd.DataFrame(
        df, columns=list(filter(lambda x: x != "UTILIZATION", df.columns.tolist()))
    )
    y = pd.Series(df["UTILIZATION"], name="UTILIZATION")
    fairness_info = {
        "favorable_labels": [[10, 10000]],
        "protected_attributes": [
            {"feature": "RACE", "reference_group": ["White"]},
        ],
    }

    return X, y, fairness_info


def fetch_meps_panel19_fy2015_df(preprocess=False):
    """
    Fetch a subset of the `MEPS`_ dataset from aif360 and add fairness info.

    It contains information collected on a nationally representative sample
    of the civilian noninstitutionalized population of the United States,
    specifically reported medical expenditures and civilian demographics.
    This dataframe corresponds to data from panel 19 from the year 2015.
    Without preprocessing, the dataframe contains 15830 rows and 138 columns.
    (With preprocessing and restriction to panel 19, the dataframe contains
    15830 rows and 138 columns.)
    There is one protected attribute, race, and the disparate impact is 0.5.
    The data includes only numeric columns, with no missing value.

    Note: in order to use this dataset, be sure to follow the instructions
    found in the `AIF360 documentation`_ and accept the corresponding license agreement.

    .. _`MEPS`:  https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181
    .. _`AIF360 documentation`: https://github.com/Trusted-AI/AIF360/tree/master/aif360/data/raw/meps

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      encode protected attribute in X corresponding to race as 0 or 1
      to indicate privileged groups;
      encode labels in y as 0 or 1 to indicate faborable outcomes;
      rename columns that are panel or round-specific;
      drop columns such as ID columns that are not relevant to the task at hand;
      and drop rows where features are unknown.

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
    if preprocess:
        dataset = aif360.datasets.MEPSDataset19()
        X, y = lale.lib.aif360.util.dataset_to_pandas(dataset)
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "RACE", "reference_group": [1]},
            ],
        }
        return X, y, fairness_info
    else:
        return fetch_meps_raw_df(Panel.PANEL19, FiscalYear.FY2015)


# def fetch_meps_panel20_fy2015_df(preprocess=False):

# def fetch_meps_panel21_fy2016_df(preprocess=False):
