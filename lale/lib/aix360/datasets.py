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

import logging
import os
import urllib.request
from enum import Enum

import aif360
import aif360.datasets
import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_compas,
)

import lale.datasets
import lale.datasets.openml
import lale.lib.aif360.util

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


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


def fetch_tae_df(preprocess=False):
    """
    Fetch the `tae`_ dataset from OpenML and add `fairness_info`.

    It contains information from teaching assistant (TA) evaluations.
    at the University of Wisconsin--Madison.
    The prediction task is a classification on the type
    of rating a TA receives (1=Low, 2=Medium, 3=High). Without preprocessing,
    the dataset has 151 rows and 5 columns. There is one protected
    attributes, "whether_of_not_the_ta_is_a_native_english_speaker" [sic],
    and the disparate impact of 0.45. The data
    includes both categorical and numeric columns, with no missing
    values.

    .. _`tae`: https://www.openml.org/d/48

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      encode protected attributes in X as 0 or 1 to indicate privileged group
      ("native_english_speaker");
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
        "tae", "classification", astype="pandas", preprocess=preprocess
    )
    orig_X = pd.concat([train_X, test_X]).sort_index().astype(np.float64)
    orig_y = pd.concat([train_y, test_y]).sort_index().astype(np.float64)

    if preprocess:
        native_english_speaker = pd.Series(
            orig_X["whether_of_not_the_ta_is_a_native_english_speaker_1"] == 1,
            dtype=np.float64,
        )
        dropped_X = orig_X.drop(
            labels=[
                "whether_of_not_the_ta_is_a_native_english_speaker_1",
                "whether_of_not_the_ta_is_a_native_english_speaker_2",
            ],
            axis=1,
        )
        encoded_X = dropped_X.assign(native_english_speaker=native_english_speaker)
        encoded_y = pd.Series(orig_y == 2, dtype=np.float64)
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "native_english_speaker", "reference_group": [1]},
            ],
        }
        return encoded_X, encoded_y, fairness_info
    else:
        fairness_info = {
            "favorable_labels": [3],
            "protected_attributes": [
                {
                    "feature": "whether_of_not_the_ta_is_a_native_english_speaker",
                    "reference_group": [1],
                },
            ],
        }
        return orig_X, orig_y, fairness_info


# COMPAS HELPERS
def _get_compas_filename(violent_recidivism=False):
    violent_tag = ""
    if violent_recidivism:
        violent_tag = "-violent"
    filename = f"compas-scores-two-years{violent_tag}.csv"
    return filename


def _get_compas_filepath(filename):
    directory = os.path.join(
        os.path.dirname(os.path.abspath(aif360.__file__)), "data", "raw", "compas"
    )
    return os.path.join(
        directory,
        filename,
    )


def _try_download_compas(violent_recidivism=False):
    filename = _get_compas_filename(violent_recidivism=violent_recidivism)
    filepath = _get_compas_filepath(filename)
    csv_exists = os.path.exists(filepath)
    if not csv_exists:
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/propublica/compas-analysis/master/{filename}",
            filepath,
        )


def _get_pandas_and_fairness_info_from_compas_dataset(dataset):
    X, y = lale.lib.aif360.util.dataset_to_pandas(dataset)
    assert X is not None

    fairness_info = {
        "favorable_labels": [0],
        "protected_attributes": [
            {"feature": "sex", "reference_group": [1]},
            {"feature": "race", "reference_group": [1]},
        ],
    }
    return X, y, fairness_info


def _get_dataframe_from_compas_csv(violent_recidivism=False):
    filename = _get_compas_filename(violent_recidivism=violent_recidivism)
    filepath = _get_compas_filepath(filename)
    try:
        df = pd.read_csv(filepath, index_col="id", na_values=[])
    except IOError as err:
        # In practice should not get here because of the _try_download_compas call above, but adding failure logic just in case
        logger.error("IOError: {}".format(err))
        logger.error("To use this class, please download the following file:")
        logger.error(
            "\n\thttps://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        )
        logger.error("\nand place it, as-is, in the folder:")
        logger.error("\n\t{}\n".format(os.path.abspath(os.path.dirname(filepath))))
        import sys

        sys.exit(1)
    if violent_recidivism:
        # violent recidivism dataset includes extra label column for some reason
        df = pd.DataFrame(
            df,
            columns=list(
                filter(lambda x: x != "two_year_recid.1", df.columns.tolist())
            ),
        ).sort_index()
    return df


def _perform_default_preprocessing(df):
    return df[
        (df.days_b_screening_arrest <= 30)
        & (df.days_b_screening_arrest >= -30)
        & (df.is_recid != -1)
        & (df.c_charge_degree != "O")
        & (df.score_text != "N/A")
    ]


def _perform_custom_preprocessing(df):
    """The custom pre-processing function is adapted from
    https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
    """

    df = df[
        [
            "age",
            "c_charge_degree",
            "race",
            "age_cat",
            "score_text",
            "sex",
            "priors_count",
            "days_b_screening_arrest",
            "decile_score",
            "is_recid",
            "two_year_recid",
            "c_jail_in",
            "c_jail_out",
        ]
    ]

    # Indices of data samples to keep
    ix = df["days_b_screening_arrest"] <= 30
    ix = (df["days_b_screening_arrest"] >= -30) & ix
    ix = (df["is_recid"] != -1) & ix
    ix = (df["c_charge_degree"] != "O") & ix
    ix = (df["score_text"] != "N/A") & ix
    df = df.loc[ix, :]
    df["length_of_stay"] = (
        pd.to_datetime(df["c_jail_out"]) - pd.to_datetime(df["c_jail_in"])
    ).apply(lambda x: x.days)

    # Restrict races to African-American and Caucasian
    dfcut = df.loc[
        ~df["race"].isin(["Native American", "Hispanic", "Asian", "Other"]), :
    ]

    # Restrict the features to use
    dfcutQ = dfcut[
        [
            "sex",
            "race",
            "age_cat",
            "c_charge_degree",
            "score_text",
            "priors_count",
            "is_recid",
            "two_year_recid",
            "length_of_stay",
        ]
    ].copy()

    # Quantize priors count between 0, 1-3, and >3
    def quantizePrior(x):
        if x <= 0:
            return "0"
        elif 1 <= x <= 3:
            return "1 to 3"
        else:
            return "More than 3"

    # Quantize length of stay
    def quantizeLOS(x):
        if x <= 7:
            return "<week"
        if 8 < x <= 93:
            return "<3months"
        else:
            return ">3 months"

    # Quantize length of stay
    def adjustAge(x):
        if x == "25 - 45":
            return "25 to 45"
        else:
            return x

    # Quantize score_text to MediumHigh
    def quantizeScore(x):
        if (x == "High") | (x == "Medium"):
            return "MediumHigh"
        else:
            return x

    def group_race(x):
        if x == "Caucasian":
            return 1.0
        else:
            return 0.0

    dfcutQ["priors_count"] = dfcutQ["priors_count"].apply(lambda x: quantizePrior(x))
    dfcutQ["length_of_stay"] = dfcutQ["length_of_stay"].apply(lambda x: quantizeLOS(x))
    dfcutQ["score_text"] = dfcutQ["score_text"].apply(lambda x: quantizeScore(x))
    dfcutQ["age_cat"] = dfcutQ["age_cat"].apply(lambda x: adjustAge(x))

    # Recode sex and race
    dfcutQ["sex"] = dfcutQ["sex"].replace({"Female": 1.0, "Male": 0.0})
    dfcutQ["race"] = dfcutQ["race"].apply(lambda x: group_race(x))

    features = [
        "two_year_recid",
        "sex",
        "race",
        "age_cat",
        "priors_count",
        "c_charge_degree",
    ]

    # Pass vallue to df
    df = dfcutQ[features]

    return df


def _get_pandas_and_fairness_info_from_compas_csv(violent_recidivism=False):
    df = _get_dataframe_from_compas_csv(violent_recidivism=violent_recidivism)
    # preprocessing steps performed by ProPublica team, even in the preprocess=False case
    df = _perform_default_preprocessing(df)
    X = pd.DataFrame(
        df, columns=list(filter(lambda x: x != "two_year_recid", df.columns.tolist()))
    ).sort_index()
    y = pd.Series(
        df["two_year_recid"], name="two_year_recid", dtype=np.float64
    ).sort_index()
    fairness_info = {
        "favorable_labels": [0],
        "protected_attributes": [
            {"feature": "sex", "reference_group": ["Female"]},
            {"feature": "race", "reference_group": ["Caucasian"]},
        ],
    }
    return X, y, fairness_info


def fetch_compas_df(preprocess=False):
    """
    Fetch the `compas-two-years`_ dataset, also known as ProPublica recidivism, from GitHub and add `fairness_info`.

    It contains information about individuals with a binary
    classification for recidivism, indicating whether they were
    re-arrested within two years after the first arrest. Without
    preprocessing, the dataset has 6,172 rows and 51 columns.  There
    are two protected attributes, sex and race, and the disparate
    impact is 0.75.  The data includes numeric and categorical columns, with some
    missing values.

    .. _`compas-two-years`: https://github.com/propublica/compas-analysis

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      encode protected attributes in X as 0 or 1 to indicate privileged groups
      (1 if Female or Caucasian for the corresponding sex and race columns respectively);
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
    violent_recidivism = False
    _try_download_compas(violent_recidivism=violent_recidivism)
    if preprocess:
        # Odd finding here: "Female" is a privileged class in the dataset, but the original
        # COMPAS algorithm actually predicted worse outcomes for that class after controlling
        # for other factors. Leaving it as "Female" for now (AIF360 does this by default as well)
        # but potentially worthy of revisiting.
        # See https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
        # and https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        # (hunch is that COMPAS was trained on more biased data that is not reproduced in ProPublica's dataset)
        dataset = load_preproc_data_compas()
        # above preprocessing results in a WARNING of "Missing Data: 5 rows removed from CompasDataset."
        # unclear how to resolve at the moment
        return _get_pandas_and_fairness_info_from_compas_dataset(dataset)
    else:
        return _get_pandas_and_fairness_info_from_compas_csv(
            violent_recidivism=violent_recidivism
        )


def fetch_compas_violent_df(preprocess=False):
    """
    Fetch the `compas-two-years-violent`_ dataset, also known as ProPublica violent recidivism, from GitHub and add `fairness_info`.

    It contains information about individuals with a binary
    classification for violent recidivism, indicating whether they were
    re-arrested within two years after the first arrest. Without
    preprocessing, the dataset has 4,020 rows and 51 columns.  There
    are three protected attributes, sex, race, and age, and the disparate
    impact is 0.85.  The data includes numeric and categorical columns, with some
    missing values.

    .. _`compas-two-years-violent`: https://github.com/propublica/compas-analysis

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      encode protected attributes in X as 0 or 1 to indicate privileged groups
      (1 if Female, Caucasian, or at least 25 for the corresponding sex, race, and
      age columns respectively);
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
    violent_recidivism = True
    _try_download_compas(violent_recidivism=violent_recidivism)
    if preprocess:
        # Odd finding here: "Female" is a privileged class in the dataset, but the original
        # COMPAS algorithm actually predicted worse outcomes for that class after controlling
        # for other factors. Leaving it as "Female" for now (AIF360 does this by default as well)
        # but potentially worthy of revisiting.
        # See https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
        # and https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        # (hunch is that COMPAS was trained on more biased data that is not reproduced in ProPublica's dataset)

        # Loading violent recidivism dataset using StandardDataset and default settings found in the CompasDataset
        # class since AIF360 lacks a violent recidivism dataset implementation
        df = _get_dataframe_from_compas_csv(violent_recidivism=violent_recidivism)
        default_mappings = {
            "label_maps": [{1.0: "Did recid.", 0.0: "No recid."}],
            "protected_attribute_maps": [
                {0.0: "Male", 1.0: "Female"},
                {1.0: "Caucasian", 0.0: "Not Caucasian"},
            ],
        }
        dataset = aif360.datasets.StandardDataset(
            df=df,
            label_name="two_year_recid",
            favorable_classes=[0],
            protected_attribute_names=["sex", "race"],
            privileged_classes=[[1.0], [1.0]],
            categorical_features=["age_cat", "priors_count", "c_charge_degree"],
            instance_weights_name=None,
            features_to_keep=[
                "sex",
                "age_cat",
                "race",
                "priors_count",
                "c_charge_degree",
                "two_year_recid",
            ],
            features_to_drop=[],
            na_values=[],
            custom_preprocessing=_perform_custom_preprocessing,
            metadata=default_mappings,
        )
        # above preprocessing results in a WARNING of "Missing Data: 5 rows removed from StandardDataset."
        # unclear how to resolve at the moment
        return _get_pandas_and_fairness_info_from_compas_dataset(dataset)
    else:
        return _get_pandas_and_fairness_info_from_compas_csv(
            violent_recidivism=violent_recidivism
        )


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
        "SpeedDating", "classification", astype="pandas", preprocess=preprocess
    )
    orig_X = pd.concat([train_X, test_X]).sort_index()
    orig_y = pd.concat([train_y, test_y]).sort_index()
    if preprocess:
        importance_same_race = pd.Series(
            orig_X["importance_same_race"] >= 9, dtype=np.float64
        )
        samerace = pd.Series(orig_X["samerace_1"] == 1, dtype=np.float64)
        # drop samerace-related columns
        columns_to_drop = ["samerace_0", "samerace_1"]

        # drop preprocessed columns

        def preprocessed_column_filter(x: str):
            return x.startswith("d_")

        columns_to_drop.extend(list(filter(preprocessed_column_filter, orig_X.columns)))

        # drop has-null columns
        columns_to_drop.extend(["has_null_0", "has_null_1"])

        # drop decision columns

        def decision_column_filter(x: str):
            return x.startswith("decision")

        columns_to_drop.extend(list(filter(decision_column_filter, orig_X.columns)))

        # drop field columns

        def field_column_filter(x: str):
            return x.startswith("field")

        columns_to_drop.extend(list(filter(field_column_filter, orig_X.columns)))

        # drop wave column
        columns_to_drop.append("wave")
        dropped_X = orig_X.drop(labels=columns_to_drop, axis=1)
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


def _fetch_boston_housing_df(preprocess=False):
    """
    Fetch the `Boston housing`_ dataset from sklearn and add `fairness info`.

    It contains data about housing values in the suburbs of Boston with various
    features that can be used to perform regression. Without preprocessing,
    the dataset has 506 rows and 14 columns. There is one protected attribute,
    1000(Bk - 0.63)^2 where Bk is the proportion of Blacks by town, and the disparate
    impact is 0.5. The data includes only numeric columns, with no missing values.

    Hiding dataset from public consumption based on issues described at length `here`_

    .. _`Boston housing`: https://scikit-learn.org/0.20/datasets/index.html#boston-house-prices-dataset
    .. _`here`: https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8

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
    assert train_X is not None
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


def fetch_nursery_df(preprocess=False):
    """
    Fetch the `nursery`_ dataset from OpenML and add `fairness_info`.

    It contains data gathered from applicants to public schools in
    Ljubljana, Slovenia during a competitive time period.
    Without preprocessing, the dataset has
    12960 rows and 8 columns.  There is one protected attribute, parents, and the
    disparate impact is 0.46.  The data has categorical columns (with
    numeric ones if preprocessing is applied), with no missing values.

    .. _`nursery`: https://www.openml.org/d/26

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      encode protected attributes in X as 0 or 1 to indicate privileged groups
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
        "nursery", "classification", astype="pandas", preprocess=preprocess
    )
    orig_X = pd.concat([train_X, test_X]).sort_index()
    orig_y = pd.concat([train_y, test_y]).sort_index()
    if preprocess:
        parents = pd.Series(orig_X["parents_usual"] == 0, dtype=np.float64)
        dropped_X = orig_X.drop(
            labels=[
                "parents_great_pret",
                "parents_pretentious",
                "parents_usual",
            ],
            axis=1,
        )
        encoded_X = dropped_X.assign(parents=parents)
        # orig_y == 3 corresponds to "spec_prior"
        encoded_y = pd.Series((orig_y == 3), dtype=np.float64)
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [{"feature": "parents", "reference_group": [1]}],
        }
        return encoded_X, encoded_y, fairness_info
    else:
        fairness_info = {
            "favorable_labels": ["spec_prior"],
            "protected_attributes": [
                {
                    "feature": "parents",
                    "reference_group": ["great_pret", "pretentious"],
                }
            ],
        }
        return orig_X, orig_y, fairness_info


def fetch_titanic_df(preprocess=False):
    """
    Fetch the `Titanic`_ dataset from OpenML and add `fairness_info`.

    It contains data gathered from passengers on the Titanic with a binary classification
    into "survived" or "did not survive".  Without preprocessing, the dataset has
    1309 rows and 13 columns.  There is one protected attribute, sex, and the
    disparate impact is 0.26.  The data includes both categorical and
    numeric columns, with some missing values.

    .. _`Titanic`: https://www.openml.org/d/40945

    Parameters
    ----------
    preprocess : boolean, optional, default False

      If True,
      encode protected attributes in X as 0 or 1 to indicate privileged groups;
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
        "titanic", "classification", astype="pandas", preprocess=preprocess
    )
    orig_X = pd.concat([train_X, test_X]).sort_index()
    orig_y = pd.concat([train_y, test_y]).sort_index()
    if preprocess:
        sex = pd.Series(orig_X["sex_female"] == 1, dtype=np.float64)
        columns_to_drop = ["sex_female", "sex_male"]

        # drop more columns that turn into gigantic one-hot encodings otherwise, like name and cabin

        def extra_categorical_columns_filter(c: str):
            return (
                c.startswith("name")
                or c.startswith("ticket")
                or c.startswith("cabin")
                or c.startswith("home.dest")
            )

        columns_to_drop.extend(
            list(filter(extra_categorical_columns_filter, orig_X.columns))
        )
        dropped_X = orig_X.drop(labels=columns_to_drop, axis=1)
        encoded_X = dropped_X.assign(sex=sex)
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "sex", "reference_group": [1]},
            ],
        }
        return encoded_X, orig_y, fairness_info
    else:
        fairness_info = {
            "favorable_labels": ["1"],
            "protected_attributes": [
                {"feature": "sex", "reference_group": ["female"]},
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


def _race(row):
    if (row["HISPANX"] == 2) and (
        row["RACEV2X"] == 1
    ):  # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
        return "White"
    return "Non-White"


def _get_utilization_columns(fiscal_year):
    return [
        f"OBTOTV{fiscal_year.value}",
        f"OPTOTV{fiscal_year.value}",
        f"ERTOT{fiscal_year.value}",
        f"IPNGTD{fiscal_year.value}",
        f"HHTOTD{fiscal_year.value}",
    ]


def _get_total_utilization(row, fiscal_year):
    cols = _get_utilization_columns(fiscal_year)
    return sum(list(map(lambda x: row[x], cols)))


def _should_drop_column(x, fiscal_year):
    utilization_cols = set(_get_utilization_columns(fiscal_year))
    return x in utilization_cols


def _fetch_meps_raw_df(panel, fiscal_year):
    filename = ""
    if fiscal_year == FiscalYear.FY2015:
        assert panel == Panel.PANEL19 or panel == Panel.PANEL20
        filename = "h181.csv"
    elif fiscal_year == FiscalYear.FY2016:
        assert panel == Panel.PANEL21
        filename = "h192.csv"
    else:
        logger.error(f"Unexpected FiscalYear received: {fiscal_year}")
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
        logger.error("IOError: {}".format(err))
        logger.error("To use this class, please follow the instructions found here:")
        logger.error(
            "\n\t{}\n".format(
                "https://github.com/Trusted-AI/AIF360/tree/master/aif360/data/raw/meps"
            )
        )
        logger.error(
            f"\n to download and convert the data and place the final {filename} file, as-is, in the folder:"
        )
        logger.error("\n\t{}\n".format(os.path.abspath(os.path.dirname(filepath))))
        import sys

        sys.exit(1)

    df["RACEV2X"] = df.apply(lambda row: _race(row), axis=1)
    df = df.rename(columns={"RACEV2X": "RACE"})
    df = df[df["PANEL"] == panel.value]

    df["TOTEXP15"] = df.apply(
        lambda row: _get_total_utilization(row, fiscal_year), axis=1
    )
    lessE = df["TOTEXP15"] < 10.0
    df.loc[lessE, "TOTEXP15"] = 0.0
    moreE = df["TOTEXP15"] >= 10.0
    df.loc[moreE, "TOTEXP15"] = 1.0

    df = df.rename(columns={"TOTEXP15": "UTILIZATION"})
    columns_to_drop = set(
        filter(lambda x: _should_drop_column(x, fiscal_year), df.columns.tolist())
    )
    df = df[sorted(set(df.columns.tolist()) - columns_to_drop, key=df.columns.get_loc)]
    X = pd.DataFrame(
        df, columns=list(filter(lambda x: x != "UTILIZATION", df.columns.tolist()))
    ).sort_index()
    y = pd.Series(df["UTILIZATION"], name="UTILIZATION").sort_index()
    fairness_info = {
        "favorable_labels": [1],
        "protected_attributes": [
            {"feature": "RACE", "reference_group": ["White"]},
        ],
    }

    return X, y, fairness_info


def _get_pandas_and_fairness_info_from_meps_dataset(dataset):
    X, y = lale.lib.aif360.util.dataset_to_pandas(dataset)
    fairness_info = {
        "favorable_labels": [1],
        "protected_attributes": [
            {"feature": "RACE", "reference_group": [1]},
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
    Without preprocessing, the dataframe contains 16578 rows and 1825 columns.
    (With preprocessing the dataframe contains 15830 rows and 138 columns.)
    There is one protected attribute, race, and the disparate impact is 0.496
    if preprocessing is not applied and 0.490 if preprocessing is applied.
    The data includes numeric and categorical columns, with some missing values.

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
        return _get_pandas_and_fairness_info_from_meps_dataset(dataset)
    else:
        return _fetch_meps_raw_df(Panel.PANEL19, FiscalYear.FY2015)


def fetch_meps_panel20_fy2015_df(preprocess=False):
    """
    Fetch a subset of the `MEPS`_ dataset from aif360 and add fairness info.

    It contains information collected on a nationally representative sample
    of the civilian noninstitutionalized population of the United States,
    specifically reported medical expenditures and civilian demographics.
    This dataframe corresponds to data from panel 20 from the year 2015.
    Without preprocessing, the dataframe contains 18849 rows and 1825 columns.
    (With preprocessing the dataframe contains 17570 rows and 138 columns.)
    There is one protected attribute, race, and the disparate impact is 0.493
    if preprocessing is not applied and 0.488 if preprocessing is applied.
    The data includes numeric and categorical columns, with some missing values.

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
        dataset = aif360.datasets.MEPSDataset20()
        return _get_pandas_and_fairness_info_from_meps_dataset(dataset)
    else:
        return _fetch_meps_raw_df(Panel.PANEL20, FiscalYear.FY2015)


def fetch_meps_panel21_fy2016_df(preprocess=False):
    """
    Fetch a subset of the `MEPS`_ dataset from aif360 and add fairness info.

    It contains information collected on a nationally representative sample
    of the civilian noninstitutionalized population of the United States,
    specifically reported medical expenditures and civilian demographics.
    This dataframe corresponds to data from panel 20 from the year 2016.
    Without preprocessing, the dataframe contains 17052 rows and 1936 columns.
    (With preprocessing the dataframe contains 15675 rows and 138 columns.)
    There is one protected attribute, race, and the disparate impact is 0.462
    if preprocessing is not applied and 0.451 if preprocessing is applied.
    The data includes numeric and categorical columns, with some missing values.

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
        dataset = aif360.datasets.MEPSDataset21()
        return _get_pandas_and_fairness_info_from_meps_dataset(dataset)
    else:
        return _fetch_meps_raw_df(Panel.PANEL21, FiscalYear.FY2016)
