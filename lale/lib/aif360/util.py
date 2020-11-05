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

import aif360.algorithms.postprocessing
import aif360.datasets
import aif360.metrics
import numpy as np
import pandas as pd
import sklearn.metrics

import lale.datasets.data_schemas
import lale.datasets.openml
import lale.type_checking


def dataset_to_pandas(dataset, return_only="Xy"):
    """
    Return pandas representation of the AIF360 dataset.

    Parameters
    ----------
    dataset : aif360.datasets.BinaryLabelDataset

      AIF360 dataset to convert to a pandas representation.

    return_only : 'Xy', 'X', or 'y'

      Which part of features X or labels y to convert and return.

    Returns
    -------
    result : tuple

      - item 0: pandas Dataframe or None, features X

      - item 1: pandas Series or None, labels y
    """
    if "X" in return_only:
        X = pd.DataFrame(dataset.features, columns=dataset.feature_names)
        result_X = lale.datasets.data_schemas.add_schema(X)
        assert isinstance(result_X, pd.DataFrame), type(result_X)
    else:
        result_X = None
    if "y" in return_only:
        y = pd.Series(dataset.labels.ravel(), name=dataset.label_names[0])
        result_y = lale.datasets.data_schemas.add_schema(y)
        assert isinstance(result_y, pd.Series), type(result_y)
    else:
        result_y = None
    return result_X, result_y


_dataset_fairness_properties: lale.type_checking.JSON_TYPE = {
    "favorable_label": {
        "description": 'Label value which is considered favorable (i.e. "positive").',
        "type": "number",
    },
    "unfavorable_label": {
        "description": 'Label value which is considered unfavorable (i.e. "negative").',
        "type": "number",
    },
    "protected_attribute_names": {
        "description": "Subset of feature names for which fairness is desired.",
        "type": "array",
        "items": {"type": "string"},
    },
    "unprivileged_groups": {
        "description": "Representation for unprivileged group.",
        "type": "array",
        "items": {
            "description": "Map from feature names to group-indicating values.",
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
    "privileged_groups": {
        "description": "Representation for privileged group.",
        "type": "array",
        "items": {
            "description": "Map from feature names to group-indicating values.",
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
    },
}

_categorical_fairness_properties: lale.type_checking.JSON_TYPE = {
    "favorable_labels": {
        "description": 'Label values which are considered favorable (i.e. "positive").',
        "type": "array",
        "minItems": 1,
        "items": {
            "anyOf": [
                {"description": "Literal value.", "type": "string"},
                {"description": "Numerical value.", "type": "number"},
                {
                    "description": "Numeric range [a,b] from a to b inclusive.",
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"},
                },
            ]
        },
    },
    "protected_attributes": {
        "description": "Features for which fairness is desired.",
        "type": "array",
        "items": {
            "type": "object",
            "required": ["feature", "privileged_groups"],
            "properties": {
                "feature": {
                    "description": "Column name or column index.",
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                },
                "privileged_groups": {
                    "description": "Values or ranges that indicate being a member of the privileged group.",
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "anyOf": [
                            {"description": "Literal value.", "type": "string"},
                            {"description": "Numerical value.", "type": "number"},
                            {
                                "description": "Numeric range [a,b] from a to b inclusive.",
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {"type": "number"},
                            },
                        ]
                    },
                },
            },
        },
    },
}

_categorical_fairness_schema = {
    "type": "object",
    "properties": _categorical_fairness_properties,
}

_dataset_fairness_schema = {
    "type": "object",
    "properties": _dataset_fairness_properties,
}


def dataset_fairness_info(dataset):
    """
    Inspect the AIF360 dataset and return its fairness metadata as JSON.

    Parameters
    ----------
    dataset : aif360.datasets.BinaryLabelDataset

    Returns
    -------
    result : dict

      JSON data structure with fairness information.

      - favorable_label : number

          Label value which is considered favorable (i.e. "positive").

      - unfavorable_label : number

          Label value which is considered unfavorable (i.e. "negative").

      - protected_attribute_names : array **of** items : string

          Subset of feature names for which fairness is desired.

      - unprivileged_groups : array

          Representation for unprivileged group.

          - items : dict

              Map from feature names to group-indicating values.

      - privileged_groups : array

          Representation for privileged group.

          - items : dict

              Map from feature names to group-indicating values.
    """

    def attributes_to_groups(names, value_arrays):
        result = [{}]
        for i in range(len(names)):
            next_result = []
            for d in result:
                for next_v in value_arrays[i]:
                    next_d = {**d, names[i]: next_v}
                    next_result.append(next_d)
            result = next_result
        return result

    unprivileged_groups = attributes_to_groups(
        dataset.protected_attribute_names, dataset.unprivileged_protected_attributes
    )
    privileged_groups = attributes_to_groups(
        dataset.protected_attribute_names, dataset.privileged_protected_attributes
    )
    result = {
        "favorable_label": dataset.favorable_label,
        "unfavorable_label": dataset.unfavorable_label,
        "protected_attribute_names": dataset.protected_attribute_names,
        "unprivileged_groups": unprivileged_groups,
        "privileged_groups": privileged_groups,
    }
    lale.type_checking.validate_schema(result, _dataset_fairness_schema)
    return result


class _PandasToDatasetConverter:
    def __init__(self, favorable_label, unfavorable_label, protected_attribute_names):
        lale.type_checking.validate_schema(
            favorable_label, _dataset_fairness_properties["favorable_label"]
        )
        self.favorable_label = favorable_label
        lale.type_checking.validate_schema(
            unfavorable_label, _dataset_fairness_properties["unfavorable_label"]
        )
        self.unfavorable_label = unfavorable_label
        lale.type_checking.validate_schema(
            protected_attribute_names,
            _dataset_fairness_properties["protected_attribute_names"],
        )
        self.protected_attribute_names = protected_attribute_names

    def __call__(self, X, y):
        assert isinstance(X, pd.DataFrame), type(X)
        assert isinstance(y, pd.Series), type(y)
        assert X.shape[0] == y.shape[0], f"X.shape {X.shape}, y.shape {y.shape}"
        assert not X.isna().any().any(), f"X\n{X}\n"
        assert not y.isna().any().any(), f"y\n{X}\n"
        y_reindexed = pd.Series(data=y.values, index=X.index, name=y.name)
        df = pd.concat([X, y_reindexed], axis=1)
        assert df.shape[0] == X.shape[0], f"df.shape {df.shape}, X.shape {X.shape}"
        assert not df.isna().any().any(), f"df\n{df}\nX\n{X}\ny\n{y}"
        label_names = [y.name]
        result = aif360.datasets.BinaryLabelDataset(
            favorable_label=self.favorable_label,
            unfavorable_label=self.unfavorable_label,
            protected_attribute_names=self.protected_attribute_names,
            df=df,
            label_names=label_names,
        )
        return result


def _group_flag(value, groups):
    for group in groups:
        if isinstance(group, list):
            if group[0] <= value <= group[1]:
                return 1
        elif value == group:
            return 1
    return 0


def _dataframe_replace(dataframe, subst):
    new_columns = [
        subst.get(i, subst.get(name, dataframe.iloc[:, i]))
        for i, name in enumerate(dataframe.columns)
    ]
    result = pd.concat(new_columns, axis=1)
    return result


def _ensure_str(str_or_int):
    return f"f{str_or_int}" if isinstance(str_or_int, int) else str_or_int


def _ndarray_to_series(data, name, index=None, dtype=None):
    if isinstance(data, pd.Series):
        return data
    result = pd.Series(data=data, index=index, dtype=dtype, name=_ensure_str(name))
    schema = getattr(data, "json_schema", None)
    if schema is not None:
        result = lale.datasets.data_schemas.add_schema(result, schema)
    return result


def _ndarray_to_dataframe(array):
    assert len(array.shape) == 2
    column_names = None
    schema = getattr(array, "json_schema", None)
    if schema is not None:
        column_schemas = schema.get("items", {}).get("items", None)
        if isinstance(column_schemas, list):
            column_names = [s.get("description", None) for s in column_schemas]
    if column_names is None or None in column_names:
        column_names = [_ensure_str(i) for i in range(array.shape[1])]
    result = pd.DataFrame(array, columns=column_names)
    if schema is not None:
        result = lale.datasets.data_schemas.add_schema(result, schema)
    return result


class _CategoricalFairnessConverter:
    def __init__(self, favorable_labels, protected_attributes, remainder="drop"):
        lale.type_checking.validate_schema(
            favorable_labels, _categorical_fairness_properties["favorable_labels"]
        )
        assert remainder in ["drop", "passthrough"]
        self.remainder = remainder
        self.favorable_labels = favorable_labels
        lale.type_checking.validate_schema(
            protected_attributes,
            _categorical_fairness_properties["protected_attributes"],
        )
        self.protected_attributes = protected_attributes

    def __call__(self, orig_X, orig_y):
        if isinstance(orig_X, np.ndarray):
            orig_X = _ndarray_to_dataframe(orig_X)
        if isinstance(orig_y, np.ndarray):
            orig_y = _ndarray_to_series(orig_y, orig_X.shape[1])
        assert isinstance(orig_X, pd.DataFrame), type(orig_X)
        assert isinstance(orig_y, pd.Series), type(orig_y)
        assert (
            orig_X.shape[0] == orig_y.shape[0]
        ), f"orig_X.shape {orig_X.shape}, orig_y.shape {orig_y.shape}"
        protected = {}
        for prot_attr in self.protected_attributes:
            feature = prot_attr["feature"]
            groups = prot_attr["privileged_groups"]
            if isinstance(feature, str):
                column = orig_X[feature]
            else:
                column = orig_X.iloc[:, feature]
            series = column.apply(lambda v: _group_flag(v, groups))
            protected[feature] = series
        if self.remainder == "drop":
            result_X = pd.concat([protected[f] for f in protected], axis=1)
        else:
            result_X = _dataframe_replace(orig_X, protected)
        result_y = orig_y.apply(lambda v: _group_flag(v, self.favorable_labels))
        return result_X, result_y


class _BinaryLabelScorer:
    def __init__(
        self,
        metric,
        favorable_label=None,
        unfavorable_label=None,
        protected_attribute_names=None,
        unprivileged_groups=None,
        privileged_groups=None,
        favorable_labels=None,
        protected_attributes=None,
    ):
        assert hasattr(aif360.metrics.BinaryLabelDatasetMetric, metric)
        self.metric = metric
        if favorable_labels is None:
            self.cats_to_binary = None
        else:
            self.cat_info = {
                "favorable_labels": favorable_labels,
                "protected_attributes": protected_attributes,
            }
            lale.type_checking.validate_schema(self.cat_info, _dataset_fairness_schema)
            assert favorable_label is None and unfavorable_label is None
            favorable_label, unfavorable_label = 1, 0
            assert protected_attribute_names is None
            pas = protected_attributes
            protected_attribute_names = [_ensure_str(pa["feature"]) for pa in pas]
            assert unprivileged_groups is None and privileged_groups is None
            unprivileged_groups = [{_ensure_str(pa["feature"]): 0 for pa in pas}]
            privileged_groups = [{_ensure_str(pa["feature"]): 1 for pa in pas}]
            self.cats_to_binary = _CategoricalFairnessConverter(**self.cat_info)
        self.fairness_info = {
            "favorable_label": favorable_label,
            "unfavorable_label": unfavorable_label,
            "protected_attribute_names": protected_attribute_names,
            "unprivileged_groups": unprivileged_groups,
            "privileged_groups": privileged_groups,
        }
        lale.type_checking.validate_schema(self.fairness_info, _dataset_fairness_schema)
        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label, unfavorable_label, protected_attribute_names
        )

    def __call__(self, estimator, X, y):
        predicted = estimator.predict(X)
        index = X.index if isinstance(X, pd.DataFrame) else None
        y_name = y.name if isinstance(y, pd.Series) else _ensure_str(X.shape[1])
        y_pred = _ndarray_to_series(predicted, y_name, index, y.dtype)
        if self.cats_to_binary is not None:
            X, y_pred = self.cats_to_binary(X, y_pred)
        dataset_pred = self.pandas_to_dataset(X, y_pred)
        fairness_metrics = aif360.metrics.BinaryLabelDatasetMetric(
            dataset_pred,
            self.fairness_info["unprivileged_groups"],
            self.fairness_info["privileged_groups"],
        )
        method = getattr(fairness_metrics, self.metric)
        result = method()
        return result


_SCORER_DOCSTRING = """

There are two ways to construct this scorer, either with
(favorable_label, unfavorable_label, protected_attribute_names,
unprivileged_groups, privileged_groups) or with
(favorable_labels, protected_attributes).

Parameters
----------
favorable_label : number

  Label value which is considered favorable (i.e. "positive").

unfavorable_label : number

  Label value which is considered unfavorable (i.e. "negative").

protected_attribute_names : array **of** items : string

  Subset of feature names for which fairness is desired.

unprivileged_groups : array

  Representation for unprivileged group.

  - items : dict

      Map from feature names to group-indicating values.

privileged_groups : array

  Representation for privileged group.

  - items : dict

      Map from feature names to group-indicating values.

favorable_labels : array of union

  Label values which are considered favorable (i.e. "positive").

  - string

      Literal value

  - number

      Numerical value

  - array of number, >= 2 items, <= 2 items

      Numeric range [a,b] from a to b inclusive.

protected_attributes : array of dict

  Features for which fairness is desired.

  - feature : string or integer

      Column name or column index.

  - privileged_groups : array of union

      Values or ranges that indicate being a member of the privileged group.

      - string

          Literal value

      - number

          Numerical value

      - array of number, >= 2 items, <= 2 items

          Numeric range [a,b] from a to b inclusive.

Returns
-------
result : callable

  Scorer that takes three arguments (estimator, X, y) and returns score.
"""


class _AccuracyAndDisparateImpact:
    def __init__(
        self,
        favorable_label=None,
        unfavorable_label=None,
        protected_attribute_names=None,
        unprivileged_groups=None,
        privileged_groups=None,
        favorable_labels=None,
        protected_attributes=None,
    ):
        self.accuracy_scorer = sklearn.metrics.make_scorer(
            sklearn.metrics.accuracy_score
        )
        self.disparate_impact_scorer = disparate_impact(
            favorable_label,
            unfavorable_label,
            protected_attribute_names,
            unprivileged_groups,
            privileged_groups,
            favorable_labels,
            protected_attributes,
        )

    def __call__(self, estimator, X, y):
        disp_impact = self.disparate_impact_scorer(estimator, X, y)
        if np.isnan(disp_impact):  # empty privileged or unprivileged groups
            return np.NAN
        accuracy = self.accuracy_scorer(estimator, X, y)
        assert 0.0 <= accuracy <= 1.0 and 0.0 <= disp_impact, (accuracy, disp_impact)
        if disp_impact <= 1.0:
            symmetric_impact = disp_impact
        else:
            symmetric_impact = 1.0 / disp_impact
        disp_impact_treshold = 0.9  # impact above threshold is considered fair
        if symmetric_impact < disp_impact_treshold:
            scaling_factor = symmetric_impact / disp_impact_treshold
        else:
            scaling_factor = 1.0
        scaling_hardness = 4.0  # higher hardness yields result closer to 0 when unfair
        result = accuracy * scaling_factor ** scaling_hardness
        assert 0.0 <= result <= accuracy <= 1.0, (result, accuracy)
        assert symmetric_impact >= 0.9 or result < accuracy
        return result


def accuracy_and_disparate_impact(
    favorable_label=None,
    unfavorable_label=None,
    protected_attribute_names=None,
    unprivileged_groups=None,
    privileged_groups=None,
    favorable_labels=None,
    protected_attributes=None,
):
    """Create a scikit-learn compatible combined scorer for `accuracy`_ and `disparate impact`_ given the fairness info.

.. _`accuracy`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
.. _`disparate impact`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.disparate_impact"""
    return _AccuracyAndDisparateImpact(
        favorable_label,
        unfavorable_label,
        protected_attribute_names,
        unprivileged_groups,
        privileged_groups,
        favorable_labels,
        protected_attributes,
    )


accuracy_and_disparate_impact.__doc__ = (
    str(accuracy_and_disparate_impact.__doc__) + _SCORER_DOCSTRING
)


def disparate_impact(
    favorable_label=None,
    unfavorable_label=None,
    protected_attribute_names=None,
    unprivileged_groups=None,
    privileged_groups=None,
    favorable_labels=None,
    protected_attributes=None,
):
    """Create a scikit-learn compatible `disparate impact`_ scorer given the fairness info.

.. _`disparate impact`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.disparate_impact"""
    return _BinaryLabelScorer(
        "disparate_impact",
        favorable_label,
        unfavorable_label,
        protected_attribute_names,
        unprivileged_groups,
        privileged_groups,
        favorable_labels,
        protected_attributes,
    )


disparate_impact.__doc__ = str(disparate_impact.__doc__) + _SCORER_DOCSTRING


class _R2AndDisparateImpact:
    def __init__(
        self,
        favorable_label=None,
        unfavorable_label=None,
        protected_attribute_names=None,
        unprivileged_groups=None,
        privileged_groups=None,
        favorable_labels=None,
        protected_attributes=None,
    ):
        self.r2_scorer = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)
        self.disparate_impact_scorer = disparate_impact(
            favorable_label,
            unfavorable_label,
            protected_attribute_names,
            unprivileged_groups,
            privileged_groups,
            favorable_labels,
            protected_attributes,
        )

    def __call__(self, estimator, X, y):
        disp_impact = self.disparate_impact_scorer(estimator, X, y)
        if np.isnan(disp_impact):  # empty privileged or unprivileged groups
            return np.NAN
        r2 = self.r2_scorer(estimator, X, y)
        assert r2 <= 1.0 and 0.0 <= disp_impact, (r2, disp_impact)
        if disp_impact <= 1.0:
            symmetric_impact = disp_impact
        else:
            symmetric_impact = 1.0 / disp_impact
        disp_impact_treshold = 0.9  # impact above threshold is considered fair
        if symmetric_impact < disp_impact_treshold:
            scaling_factor = symmetric_impact / disp_impact_treshold
        else:
            scaling_factor = 1.0
        scaling_hardness = 4.0  # higher hardness yields result closer to 0 when unfair
        positive_r2 = 1.0 - r2
        scaled_r2 = positive_r2 / scaling_factor ** scaling_hardness
        result = 1.0 - scaled_r2
        assert result <= r2 <= 1.0, (result, r2)
        assert symmetric_impact >= 0.9 or result < r2
        return result


def r2_and_disparate_impact(
    favorable_label=None,
    unfavorable_label=None,
    protected_attribute_names=None,
    unprivileged_groups=None,
    privileged_groups=None,
    favorable_labels=None,
    protected_attributes=None,
):
    """Create a scikit-learn compatible combined scorer for `R2 score`_ and `disparate impact`_ given the fairness info.

.. _`R2 score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
.. _`disparate impact`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.disparate_impact"""
    return _R2AndDisparateImpact(
        favorable_label,
        unfavorable_label,
        protected_attribute_names,
        unprivileged_groups,
        privileged_groups,
        favorable_labels,
        protected_attributes,
    )


r2_and_disparate_impact.__doc__ = (
    str(r2_and_disparate_impact.__doc__) + _SCORER_DOCSTRING
)


def statistical_parity_difference(
    favorable_label=None,
    unfavorable_label=None,
    protected_attribute_names=None,
    unprivileged_groups=None,
    privileged_groups=None,
    favorable_labels=None,
    protected_attributes=None,
):
    """Create a scikit-learn compatible `statistical parity difference`_ scorer given the fairness info.

.. _`statistical parity difference`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.statistical_parity_difference"""
    return _BinaryLabelScorer(
        "statistical_parity_difference",
        favorable_label,
        unfavorable_label,
        protected_attribute_names,
        unprivileged_groups,
        privileged_groups,
        favorable_labels,
        protected_attributes,
    )


statistical_parity_difference.__doc__ = (
    str(statistical_parity_difference.__doc__) + _SCORER_DOCSTRING
)


_postprocessing_base_hyperparams = {
    "estimator": {
        "description": "Nested supervised learning operator for which to mitigate fairness.",
        "laleType": "operator",
    },
    "favorable_label": _dataset_fairness_properties["favorable_label"],
    "unfavorable_label": _dataset_fairness_properties["unfavorable_label"],
    "protected_attribute_names": _dataset_fairness_properties[
        "protected_attribute_names"
    ],
}


class _BasePostprocessingImpl:
    def __init__(
        self,
        mitigator,
        estimator,
        favorable_label,
        unfavorable_label,
        protected_attribute_names,
    ):
        self.mitigator = mitigator
        self.estimator = estimator
        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label, unfavorable_label, protected_attribute_names
        )
        self.y_dtype = None
        self.y_name = None

    def fit(self, X, y):
        self.y_dtype = y.dtype
        self.y_name = y.name
        y_true = y
        self.estimator = self.estimator.fit(X, y_true)
        predicted = self.estimator.predict(X)
        y_pred = _ndarray_to_series(predicted, self.y_name, X.index, self.y_dtype)
        dataset_true = self.pandas_to_dataset(X, y_true)
        dataset_pred = self.pandas_to_dataset(X, y_pred)
        self.mitigator = self.mitigator.fit(dataset_true, dataset_pred)
        return self

    def predict(self, X):
        predicted = self.estimator.predict(X)
        y_pred = _ndarray_to_series(predicted, self.y_name, X.index, self.y_dtype)
        dataset_pred = self.pandas_to_dataset(X, y_pred)
        dataset_out = self.mitigator.predict(dataset_pred)
        _, y_out = dataset_to_pandas(dataset_out, return_only="y")
        return y_out


_numeric_supervised_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "type": "array",
            "items": {"type": "number"},
        },
    },
}

_numeric_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_numeric_output_predict_schema = {
    "description": "Predicted class label per sample.",
    "type": "array",
    "items": {"type": "number"},
}


def fetch_creditg_df():
    (orig_train_X, train_y), (orig_test_X, test_y) = lale.datasets.openml.fetch(
        "credit-g", "classification", astype="pandas"
    )

    def replace_protected_attr(orig_X):
        sex = pd.Series(
            (orig_X["personal_status_male div/sep"] == 1.0)
            | (orig_X["personal_status_male mar/wid"] == 1.0)
            | (orig_X["personal_status_male single"] == 1.0),
            dtype=np.float64,
        )
        age = pd.Series(orig_X["age"] > 25, dtype=np.float64)
        dropped = orig_X.drop(
            labels=[
                "personal_status_female div/dep/mar",
                "personal_status_male div/sep",
                "personal_status_male mar/wid",
                "personal_status_male single",
            ],
            axis=1,
        )
        added = dropped.assign(sex=sex, age=age)
        return added

    train_X = replace_protected_attr(orig_train_X)
    test_X = replace_protected_attr(orig_test_X)
    return (train_X, train_y), (test_X, test_y)


def fetch_adult_df(test_size=0.3):
    (orig_train_X, train_y), (orig_test_X, test_y) = lale.datasets.openml.fetch(
        "adult", "classification", astype="pandas", test_size=test_size
    )

    def replace_protected_attr(orig_X):
        sex = pd.Series(orig_X["sex_Male"] == 1.0, dtype=np.float64)
        race = pd.Series(orig_X["race_White"] == 1.0, dtype=np.float64)
        dropped = orig_X.drop(
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
        added = dropped.assign(sex=sex, race=race)
        return added

    train_X = replace_protected_attr(orig_train_X)
    test_X = replace_protected_attr(orig_test_X)
    assert not train_X.isna().any().any()
    assert not train_y.isna().any().any()
    assert not test_X.isna().any().any()
    assert not test_y.isna().any().any()
    return (train_X, train_y), (test_X, test_y)
