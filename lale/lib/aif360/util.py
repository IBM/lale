# Copyright 2019, 2020, 2021 IBM Corporation
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
from typing import Optional, Tuple

import aif360.algorithms.postprocessing
import aif360.datasets
import aif360.metrics
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection

import lale.datasets.data_schemas
import lale.datasets.openml
import lale.lib.lale
import lale.operators
import lale.type_checking
from lale.datasets.data_schemas import add_schema_adjusting_n_rows

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def dataset_to_pandas(
    dataset, return_only="Xy"
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
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
        "minItems": 1,
        "items": {
            "type": "object",
            "required": ["feature", "reference_group"],
            "properties": {
                "feature": {
                    "description": "Column name or column index.",
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                },
                "reference_group": {
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
                "monitored_group": {
                    "description": "Values or ranges that indicate being a member of the unprivileged group.",
                    "anyOf": [
                        {
                            "description": "If `monitored_group` is not explicitly specified, consider any values not captured by `reference_group` as monitored.",
                            "enum": [None],
                        },
                        {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "anyOf": [
                                    {"description": "Literal value.", "type": "string"},
                                    {
                                        "description": "Numerical value.",
                                        "type": "number",
                                    },
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
                    ],
                    "default": None,
                },
            },
        },
    },
    "unfavorable_labels": {
        "description": 'Label values which are considered unfavorable (i.e. "negative").',
        "anyOf": [
            {
                "description": "If `unfavorable_labels` is not explicitly specified, consider any labels not captured by `favorable_labels` as unfavorable.",
                "enum": [None],
            },
            {
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
                    ],
                },
            },
        ],
        "default": None,
    },
}

FAIRNESS_INFO_SCHEMA = {
    "type": "object",
    "properties": _categorical_fairness_properties,
}


def _validate_fairness_info(
    favorable_labels, protected_attributes, unfavorable_labels, check_schema
):
    if check_schema:
        lale.type_checking.validate_schema_directly(
            {
                "favorable_labels": favorable_labels,
                "protected_attributes": protected_attributes,
                "unfavorable_labels": unfavorable_labels,
            },
            FAIRNESS_INFO_SCHEMA,
        )

    def _check_ranges(base_name, name, groups):
        for group in groups:
            if isinstance(group, list):
                if group[0] > group[1]:
                    if base_name is None:
                        logger.warning(f"range {group} in {name} has min>max")
                    else:
                        logger.warning(
                            f"range {group} in {name} of feature '{base_name}' has min>max"
                        )

    def _check_overlaps(base_name, name1, groups1, name2, groups2):
        for g1 in groups1:
            for g2 in groups2:
                overlap = False
                if isinstance(g1, list):
                    if isinstance(g2, list):
                        overlap = g1[0] <= g2[0] <= g1[1] or g1[0] <= g2[1] <= g1[1]
                    else:
                        overlap = g1[0] <= g2 <= g1[1]
                else:
                    if isinstance(g2, list):
                        overlap = g2[0] <= g1 <= g2[1]
                    else:
                        overlap = g1 == g2
                if overlap:
                    s1 = f"'{g1}'" if isinstance(g1, str) else str(g1)
                    s2 = f"'{g2}'" if isinstance(g2, str) else str(g2)
                    if base_name is None:
                        logger.warning(
                            f"overlap between {name1} and {name2} on {s1} and {s2}"
                        )
                    else:
                        logger.warning(
                            f"overlap between {name1} and {name2} of feature '{base_name}' on {s1} and {s2}"
                        )

    _check_ranges(None, "favorable labels", favorable_labels)
    if unfavorable_labels is not None:
        _check_ranges(None, "unfavorable labels", unfavorable_labels)
        _check_overlaps(
            None,
            "favorable labels",
            favorable_labels,
            "unfavorable labels",
            unfavorable_labels,
        )
    for attr in protected_attributes:
        base_name = attr["feature"]
        reference = attr["reference_group"]
        _check_ranges(base_name, "reference group", reference)
        monitored = attr.get("monitored_group", None)
        if monitored is not None:
            _check_ranges(base_name, "monitored group", monitored)
            _check_overlaps(
                base_name, "reference group", reference, "monitored group", monitored
            )


class _PandasToDatasetConverter:
    def __init__(self, favorable_label, unfavorable_label, protected_attribute_names):
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.protected_attribute_names = protected_attribute_names

    def convert(self, X, y, probas=None):
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
        if probas is not None:
            pos_ind = 1  # TODO: is this always the case?
            result.scores = probas[:, pos_ind].reshape(-1, 1)
        return result


def _ensure_str(str_or_int):
    return f"f{str_or_int}" if isinstance(str_or_int, int) else str_or_int


def _ndarray_to_series(data, name, index=None, dtype=None) -> pd.Series:
    if isinstance(data, pd.Series):
        return data
    result = pd.Series(data=data, index=index, dtype=dtype, name=_ensure_str(name))
    schema = getattr(data, "json_schema", None)
    if schema is not None:
        result = lale.datasets.data_schemas.add_schema(result, schema)
    return result


def _ndarray_to_dataframe(array) -> pd.DataFrame:
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


class _ScorerFactory:
    def __init__(
        self, metric, favorable_labels, protected_attributes, unfavorable_labels
    ):
        _validate_fairness_info(
            favorable_labels, protected_attributes, unfavorable_labels, True
        )
        if metric in ["disparate_impact", "statistical_parity_difference"]:
            unfavorable_labels = None  # not used and may confound AIF360
        if hasattr(aif360.metrics.BinaryLabelDatasetMetric, metric):
            self.kind = "BinaryLabelDatasetMetric"
        elif hasattr(aif360.metrics.ClassificationMetric, metric):
            self.kind = "ClassificationMetric"
        else:
            raise ValueError(f"unknown metric {metric}")
        self.metric = metric
        self.fairness_info = {
            "favorable_labels": favorable_labels,
            "protected_attributes": protected_attributes,
            "unfavorable_labels": unfavorable_labels,
        }

        from lale.lib.aif360 import ProtectedAttributesEncoder

        self.prot_attr_enc = ProtectedAttributesEncoder(
            **self.fairness_info,
            remainder="drop",
            return_X_y=True,
        )
        pas = protected_attributes
        self.unprivileged_groups = [{_ensure_str(pa["feature"]): 0 for pa in pas}]
        self.privileged_groups = [{_ensure_str(pa["feature"]): 1 for pa in pas}]

        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label=1,
            unfavorable_label=0,
            protected_attribute_names=[_ensure_str(pa["feature"]) for pa in pas],
        )

    def score_data(self, y_true=None, y_pred=None, X=None):
        assert y_pred is not None
        assert X is not None
        y_pred_orig = y_pred
        if not isinstance(y_pred, pd.Series):
            assert y_true is not None
            y_pred = _ndarray_to_series(
                y_pred,
                y_true.name
                if isinstance(y_true, pd.Series)
                else _ensure_str(X.shape[1]),
                X.index if isinstance(X, pd.DataFrame) else None,
                y_pred.dtype,
            )
        encoded_X, y_pred = self.prot_attr_enc.transform(X, y_pred)
        try:
            dataset_pred = self.pandas_to_dataset.convert(encoded_X, y_pred)
        except ValueError as e:
            raise ValueError(
                "The data has unexpected labels given the fairness info: "
                f"favorable labels {self.fairness_info['favorable_labels']}, "
                f"unfavorable labels {self.fairness_info['unfavorable_labels']}, "
                f"unique values in y_pred {set(y_pred_orig)}."
            ) from e
        if self.kind == "BinaryLabelDatasetMetric":
            fairness_metrics = aif360.metrics.BinaryLabelDatasetMetric(
                dataset_pred, self.unprivileged_groups, self.privileged_groups
            )
        else:
            assert self.kind == "ClassificationMetric"
            assert y_true is not None
            if not isinstance(y_true, pd.Series):
                y_true = _ndarray_to_series(
                    y_true, y_pred.name, y_pred.index, y_pred_orig.dtype
                )
            _, y_true = self.prot_attr_enc.transform(X, y_true)
            dataset_true = self.pandas_to_dataset.convert(encoded_X, y_true)
            fairness_metrics = aif360.metrics.ClassificationMetric(
                dataset_true,
                dataset_pred,
                self.unprivileged_groups,
                self.privileged_groups,
            )
        method = getattr(fairness_metrics, self.metric)
        result = method()
        if np.isnan(result) or not np.isfinite(result):
            if 0 == fairness_metrics.num_positives(privileged=True):
                logger.warning("there are 0 positives in the privileged group")
            if 0 == fairness_metrics.num_positives(privileged=False):
                logger.warning("there are 0 positives in the unprivileged group")
            if 0 == fairness_metrics.num_instances(privileged=True):
                logger.warning("there are 0 instances in the privileged group")
            if 0 == fairness_metrics.num_instances(privileged=False):
                logger.warning("there are 0 instances in the unprivileged group")
            logger.warning(
                f"The metric {self.metric} is ill-defined and returns {result}. Check your fairness configuration. The set of predicted labels is {set(y_pred_orig)}."
            )
        return result

    def score_estimator(self, estimator, X, y):
        return self.score_data(y_true=y, y_pred=estimator.predict(X), X=X)

    def __call__(self, estimator, X, y):
        return self.score_estimator(estimator, X, y)


_SCORER_DOCSTRING_ARGS = """

    Parameters
    ----------
    favorable_labels : array of union

      Label values which are considered favorable (i.e. "positive").

      - string

          Literal value

      - *or* number

          Numerical value

      - *or* array of numbers, >= 2 items, <= 2 items

          Numeric range [a,b] from a to b inclusive.

    protected_attributes : array of dict

      Features for which fairness is desired.

      - feature : string or integer

          Column name or column index.

      - reference_group : array of union

          Values or ranges that indicate being a member of the privileged group.

          - string

              Literal value

          - *or* number

              Numerical value

          - *or* array of numbers, >= 2 items, <= 2 items

              Numeric range [a,b] from a to b inclusive.

      - monitored_group : union type, default None

          Values or ranges that indicate being a member of the unprivileged group.

          - None

              If `monitored_group` is not explicitly specified, consider any values not captured by `reference_group` as monitored.

          - *or* array of union

            - string

                Literal value

            - *or* number

                Numerical value

            - *or* array of numbers, >= 2 items, <= 2 items

                Numeric range [a,b] from a to b inclusive.

    unfavorable_labels : union type, default None

      Label values which are considered unfavorable (i.e. "negative").

      - None

          If `unfavorable_labels` is not explicitly specified, consider any labels not captured by `favorable_labels` as unfavorable.

      - *or* array of union

        - string

            Literal value

        - *or* number

            Numerical value

        - *or* array of numbers, >= 2 items, <= 2 items

            Numeric range [a,b] from a to b inclusive."""

_SCORER_DOCSTRING_RETURNS = """

    Returns
    -------
    result : callable

      Scorer that takes three arguments ``(estimator, X, y)`` and returns a
      scalar number.  Furthermore, besides being callable, the returned object
      also has two methods, ``score_data(y_true, y_pred, X)`` for evaluating
      datasets and ``score_estimator(estimator, X, y)`` for evaluating
      estimators.
"""

_SCORER_DOCSTRING = _SCORER_DOCSTRING_ARGS + _SCORER_DOCSTRING_RETURNS

_COMBINED_SCORER_DOCSTRING = (
    _SCORER_DOCSTRING_ARGS
    + """

    fairness_weight : number, >=0, <=1, default=0.5

      At the default weight of 0.5, the two metrics contribute equally to the combined result. Above 0.5, fairness influences the combination more, and below 0.5, fairness influences the combination less. In the extreme, at 1, the outcome is only determined by fairness, and at 0, the outcome ignores fairness.
"""
    + _SCORER_DOCSTRING_RETURNS
)


class _AccuracyAndDisparateImpact:
    def __init__(
        self,
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
        fairness_weight,
    ):
        if fairness_weight < 0.0 or fairness_weight > 1.0:
            logger.warning(
                f"invalid fairness_weight {fairness_weight}, setting it to 0.5"
            )
            fairness_weight = 0.5
        self.accuracy_scorer = sklearn.metrics.make_scorer(
            sklearn.metrics.accuracy_score
        )
        self.symm_di_scorer = symmetric_disparate_impact(
            favorable_labels, protected_attributes, unfavorable_labels
        )
        self.fairness_weight = fairness_weight

    def _combine(self, accuracy, symm_di):
        if accuracy < 0.0 or accuracy > 1.0:
            logger.warning(f"invalid accuracy {accuracy}, setting it to zero")
            accuracy = 0.0
        if symm_di < 0.0 or symm_di > 1.0 or np.isinf(symm_di) or np.isnan(symm_di):
            logger.warning(f"invalid symm_di {symm_di}, setting it to zero")
            symm_di = 0.0
        result = (1 - self.fairness_weight) * accuracy + self.fairness_weight * symm_di
        if result < 0.0 or result > 1.0:
            logger.warning(
                f"unexpected result {result} for accuracy {accuracy} and symm_di {symm_di}"
            )
        return result

    def score_data(self, y_true=None, y_pred=None, X=None):
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        symm_di = self.symm_di_scorer.score_data(y_true, y_pred, X)
        return self._combine(accuracy, symm_di)

    def score_estimator(self, estimator, X, y):
        accuracy = self.accuracy_scorer(estimator, X, y)
        symm_di = self.symm_di_scorer.score_estimator(estimator, X, y)
        return self._combine(accuracy, symm_di)

    def __call__(self, estimator, X, y):
        return self.score_estimator(estimator, X, y)


def accuracy_and_disparate_impact(
    favorable_labels, protected_attributes, unfavorable_labels=None, fairness_weight=0.5
):
    """
    Create a scikit-learn compatible combined scorer for `accuracy`_
    and `symmetric disparate impact`_ given the fairness info.
    The scorer is suitable for classification problems,
    with higher resulting scores indicating better outcomes.
    The result is a linear combination of accuracy and
    symmetric disparate impact, and is between 0 and 1.
    This metric can be used as the `scoring` argument
    of an optimizer such as `Hyperopt`_, as shown in this `demo`_.

    .. _`accuracy`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    .. _`symmetric disparate impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.symmetric_disparate_impact
    .. _`Hyperopt`: lale.lib.lale.hyperopt.html#lale.lib.lale.hyperopt.Hyperopt
    .. _`demo`: https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/demo_aif360.ipynb"""
    return _AccuracyAndDisparateImpact(
        favorable_labels, protected_attributes, unfavorable_labels, fairness_weight
    )


accuracy_and_disparate_impact.__doc__ = (
    str(accuracy_and_disparate_impact.__doc__) + _COMBINED_SCORER_DOCSTRING
)


def average_odds_difference(
    favorable_labels, protected_attributes, unfavorable_labels=None
):
    r"""
    Create a scikit-learn compatible `average odds difference`_ scorer
    given the fairness info. Average of difference in false positive
    rate and true positive rate between unprivileged and privileged
    groups.

    .. math::
        \tfrac{1}{2}\left[(\text{FPR}_{D = \text{unprivileged}} - \text{FPR}_{D = \text{privileged}}) + (\text{TPR}_{D = \text{unprivileged}} - \text{TPR}_{D = \text{privileged}})\right]

    The ideal value of this metric is 0. A value of <0 implies higher
    benefit for the privileged group and a value >0 implies higher
    benefit for the unprivileged group. Fairness for this metric is
    between -0.1 and 0.1.

    .. _`average odds difference`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html#aif360.metrics.ClassificationMetric.average_odds_difference"""
    return _ScorerFactory(
        "average_odds_difference",
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
    )


average_odds_difference.__doc__ = (
    str(average_odds_difference.__doc__) + _SCORER_DOCSTRING
)


def disparate_impact(favorable_labels, protected_attributes, unfavorable_labels=None):
    r"""
    Create a scikit-learn compatible `disparate_impact`_ scorer given
    the fairness info (`Feldman et al. 2015`_). Ratio of rate of
    favorable outcome for the unprivileged group to that of the
    privileged group.

    .. math::
        \frac{\text{Pr}(Y = \text{favorable} | D = \text{unprivileged})}
        {\text{Pr}(Y = \text{favorable} | D = \text{privileged})}

    In the case of multiple protected attributes,
    `D=privileged` means all protected attributes of the sample have
    corresponding privileged values in the reference group, and
    `D=unprivileged` means all protected attributes of the sample have
    corresponding unprivileged values in the monitored group.
    The ideal value of this metric is 1. A value <1 implies a higher
    benefit for the privileged group and a value >1 implies a higher
    benefit for the unprivileged group. Fairness for this metric is
    between 0.8 and 1.25.

    .. _`disparate_impact`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.disparate_impact
    .. _`Feldman et al. 2015`: https://doi.org/10.1145/2783258.2783311"""
    return _ScorerFactory(
        "disparate_impact", favorable_labels, protected_attributes, unfavorable_labels
    )


disparate_impact.__doc__ = str(disparate_impact.__doc__) + _SCORER_DOCSTRING


def equal_opportunity_difference(
    favorable_labels, protected_attributes, unfavorable_labels=None
):
    r"""
    Create a scikit-learn compatible `equal opportunity difference`_
    scorer given the fairness info. Difference of true positive rates
    between the unprivileged and the privileged groups. The true
    positive rate is the ratio of true positives to the total number
    of actual positives for a given group.

    .. math::
        \text{TPR}_{D = \text{unprivileged}} - \text{TPR}_{D = \text{privileged}}

    The ideal value is 0. A value of <0 implies disparate benefit for the
    privileged group and a value >0 implies disparate benefit for the
    unprivileged group. Fairness for this metric is between -0.1 and 0.1.

    .. _`equal opportunity difference`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html#aif360.metrics.ClassificationMetric.equal_opportunity_difference"""
    return _ScorerFactory(
        "equal_opportunity_difference",
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
    )


equal_opportunity_difference.__doc__ = (
    str(equal_opportunity_difference.__doc__) + _SCORER_DOCSTRING
)


class _R2AndDisparateImpact:
    def __init__(
        self,
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
        fairness_weight,
    ):
        if fairness_weight < 0.0 or fairness_weight > 1.0:
            logger.warning(
                f"invalid fairness_weight {fairness_weight}, setting it to 0.5"
            )
            fairness_weight = 0.5
        self.r2_scorer = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)
        self.symm_di_scorer = symmetric_disparate_impact(
            favorable_labels, protected_attributes, unfavorable_labels
        )
        self.fairness_weight = fairness_weight

    def _combine(self, r2, symm_di):
        if r2 > 1.0:
            logger.warning(f"invalid r2 {r2}, setting it to float min")
            r2 = np.finfo(np.float32).min
        if symm_di < 0.0 or symm_di > 1.0 or np.isinf(symm_di) or np.isnan(symm_di):
            logger.warning(f"invalid symm_di {symm_di}, setting it to zero")
            symm_di = 0.0
        pos_r2 = 1 / (2.0 - r2)
        result = (1 - self.fairness_weight) * pos_r2 + self.fairness_weight * symm_di
        if result < 0.0 or result > 1.0:
            logger.warning(
                f"unexpected result {result} for r2 {r2} and symm_di {symm_di}"
            )
        return result

    def score_data(self, y_true=None, y_pred=None, X=None):
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        symm_di = self.symm_di_scorer.score_data(y_true, y_pred, X)
        return self._combine(r2, symm_di)

    def score_estimator(self, estimator, X, y):
        r2 = self.r2_scorer(estimator, X, y)
        symm_di = self.symm_di_scorer.score_estimator(estimator, X, y)
        return self._combine(r2, symm_di)

    def __call__(self, estimator, X, y):
        return self.score_estimator(estimator, X, y)


def r2_and_disparate_impact(
    favorable_labels, protected_attributes, unfavorable_labels=None, fairness_weight=0.5
):
    """
    Create a scikit-learn compatible combined scorer for `R2 score`_
    and `symmetric disparate impact`_ given the fairness info.
    The scorer is suitable for regression problems,
    with higher resulting scores indicating better outcomes.
    It first scales R2, which might be negative, to be between 0 and 1.
    Then, the result is a linear combination of the scaled R2 and
    symmetric disparate impact, and is also between 0 and 1.
    This metric can be used as the `scoring` argument
    of an optimizer such as `Hyperopt`_.

    .. _`R2 score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
    .. _`symmetric disparate impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.symmetric_disparate_impact
    .. _`Hyperopt`: lale.lib.lale.hyperopt.html#lale.lib.lale.hyperopt.Hyperopt"""
    return _R2AndDisparateImpact(
        favorable_labels, protected_attributes, unfavorable_labels, fairness_weight
    )


r2_and_disparate_impact.__doc__ = (
    str(r2_and_disparate_impact.__doc__) + _COMBINED_SCORER_DOCSTRING
)


def statistical_parity_difference(
    favorable_labels, protected_attributes, unfavorable_labels=None
):
    r"""
    Create a scikit-learn compatible `statistical parity difference`_
    scorer given the fairness info. Difference of the rate of
    favorable outcomes received by the unprivileged group to the
    privileged group.

    .. math::
        \text{Pr}(Y = \text{favorable} | D = \text{unprivileged})
        - \text{Pr}(Y = \text{favorable} | D = \text{privileged})

    The ideal value of this metric is 0. A value of <0 implies higher
    benefit for the privileged group and a value >0 implies higher
    benefit for the unprivileged group. Fairness for this metric is
    between -0.1 and 0.1. For a discussion of potential issues with
    this metric see (`Dwork et al. 2012`_).

    .. _`statistical parity difference`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.BinaryLabelDatasetMetric.html#aif360.metrics.BinaryLabelDatasetMetric.statistical_parity_difference
    .. _`Dwork et al. 2012`: https://doi.org/10.1145/2090236.2090255"""
    return _ScorerFactory(
        "statistical_parity_difference",
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
    )


statistical_parity_difference.__doc__ = (
    str(statistical_parity_difference.__doc__) + _SCORER_DOCSTRING
)


class _SymmetricDisparateImpact:
    def __init__(self, favorable_labels, protected_attributes, unfavorable_labels):
        self.disparate_impact_scorer = disparate_impact(
            favorable_labels, protected_attributes, unfavorable_labels
        )

    def _make_symmetric(self, disp_impact):
        if np.isnan(disp_impact):  # empty privileged or unprivileged groups
            return disp_impact
        if disp_impact <= 1.0:
            return disp_impact
        return 1.0 / disp_impact

    def score_data(self, y_true=None, y_pred=None, X=None):
        disp_impact = self.disparate_impact_scorer.score_data(y_true, y_pred, X)
        return self._make_symmetric(disp_impact)

    def score_estimator(self, estimator, X, y):
        disp_impact = self.disparate_impact_scorer.score_estimator(estimator, X, y)
        return self._make_symmetric(disp_impact)

    def __call__(self, estimator, X, y):
        return self.score_estimator(estimator, X, y)


def symmetric_disparate_impact(
    favorable_labels, protected_attributes, unfavorable_labels=None
):
    """
    Create a scikit-learn compatible scorer for symmetric `disparate impact`_ given the fairness info.
    For disparate impact <= 1.0, return that value, otherwise return
    its inverse.  The result is between 0 and 1.  The higher this
    metric, the better, and the ideal value is 1.  A value <1 implies
    that either the privileged group or the unprivileged group is
    receiving a disparate benefit.

    .. _`disparate impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.disparate_impact"""
    return _SymmetricDisparateImpact(
        favorable_labels, protected_attributes, unfavorable_labels
    )


symmetric_disparate_impact.__doc__ = (
    str(symmetric_disparate_impact.__doc__) + _SCORER_DOCSTRING
)


def theil_index(favorable_labels, protected_attributes, unfavorable_labels=None):
    r"""
    Create a scikit-learn compatible `Theil index`_ scorer given the
    fairness info (`Speicher et al. 2018`_). Generalized entropy of
    benefit for all individuals in the dataset, with alpha=1. Measures
    the inequality in benefit allocation for individuals.  With
    :math:`b_i = \hat{y}_i - y_i + 1`:

    .. math::
        \mathcal{E}(\alpha) = \begin{cases}
          \frac{1}{n \alpha (\alpha-1)}\sum_{i=1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right],& \alpha \ne 0, 1,\\
          \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu},& \alpha=1,\\
          -\frac{1}{n}\sum_{i=1}^n\ln\frac{b_{i}}{\mu},& \alpha=0.
        \end{cases}

    A value of 0 implies perfect fairness. Fairness is indicated by
    lower scores, higher scores are problematic.

    .. _`Theil index`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html#aif360.metrics.ClassificationMetric.theil_index
    .. _`Speicher et al. 2018`: https://doi.org/10.1145/3219819.3220046"""
    return _ScorerFactory(
        "theil_index", favorable_labels, protected_attributes, unfavorable_labels
    )


theil_index.__doc__ = str(theil_index.__doc__) + _SCORER_DOCSTRING


class _BaseInEstimatorImpl:
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
        redact,
        preparation,
        mitigator,
    ):
        _validate_fairness_info(
            favorable_labels, protected_attributes, unfavorable_labels, False
        )
        self.favorable_labels = favorable_labels
        self.protected_attributes = protected_attributes
        self.unfavorable_labels = unfavorable_labels
        self.redact = redact
        if preparation is None:
            preparation = lale.lib.lale.NoOp
        self.preparation = preparation
        self.mitigator = mitigator

    def _prep_and_encode(self, X, y=None):
        prepared_X = self.redact_and_prep.transform(X, y)
        encoded_X, encoded_y = self.prot_attr_enc.transform(X, y)
        combined_attribute_names = list(prepared_X.columns) + [
            name for name in encoded_X.columns if name not in prepared_X.columns
        ]
        combined_columns = [
            encoded_X[name] if name in encoded_X else prepared_X[name]
            for name in combined_attribute_names
        ]
        combined_X = pd.concat(combined_columns, axis=1)
        result = self.pandas_to_dataset.convert(combined_X, encoded_y)
        return result

    def _decode(self, y):
        assert isinstance(y, pd.Series)
        assert len(self.favorable_labels) == 1 and len(self.not_favorable_labels) == 1
        favorable, not_favorable = (
            self.favorable_labels[0],
            self.not_favorable_labels[0],
        )
        result = y.map(lambda label: favorable if label == 1 else not_favorable)
        return result

    def fit(self, X, y):
        from lale.lib.aif360 import ProtectedAttributesEncoder, Redacting

        fairness_info = {
            "favorable_labels": self.favorable_labels,
            "protected_attributes": self.protected_attributes,
            "unfavorable_labels": self.unfavorable_labels,
        }
        redacting = Redacting(**fairness_info) if self.redact else lale.lib.lale.NoOp
        trainable_redact_and_prep = redacting >> self.preparation
        assert isinstance(trainable_redact_and_prep, lale.operators.TrainablePipeline)
        self.redact_and_prep = trainable_redact_and_prep.fit(X, y)
        self.prot_attr_enc = ProtectedAttributesEncoder(
            **fairness_info,
            remainder="drop",
            return_X_y=True,
        )
        prot_attr_names = [pa["feature"] for pa in self.protected_attributes]
        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label=1,
            unfavorable_label=0,
            protected_attribute_names=prot_attr_names,
        )
        encoded_data = self._prep_and_encode(X, y)
        self.mitigator.fit(encoded_data)
        self.classes_ = set(list(y))
        self.not_favorable_labels = list(
            self.classes_ - set(list(self.favorable_labels))
        )
        self.classes_ = np.array(list(self.classes_))
        return self

    def predict(self, X, **predict_params):
        encoded_data = self._prep_and_encode(X)
        result_data = self.mitigator.predict(encoded_data, **predict_params)
        _, result_y = dataset_to_pandas(result_data, return_only="y")
        decoded_y = self._decode(result_y)
        return decoded_y

    def predict_proba(self, X):
        # Note, will break for GerryFairClassifier
        encoded_data = self._prep_and_encode(X)
        result_data = self.mitigator.predict(encoded_data)
        favorable_probs = result_data.scores
        all_probs = np.hstack([1 - favorable_probs, favorable_probs])
        return all_probs


class _BasePostEstimatorImpl:
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
        estimator,
        redact,
        mitigator,
    ):
        _validate_fairness_info(
            favorable_labels, protected_attributes, unfavorable_labels, True
        )
        self.favorable_labels = favorable_labels
        self.protected_attributes = protected_attributes
        self.unfavorable_labels = unfavorable_labels
        self.estimator = estimator
        self.redact = redact
        self.mitigator = mitigator

    def _decode(self, y):
        assert isinstance(y, pd.Series)
        assert len(self.favorable_labels) == 1 and len(self.not_favorable_labels) == 1
        favorable, not_favorable = (
            self.favorable_labels[0],
            self.not_favorable_labels[0],
        )
        result = y.map(lambda label: favorable if label == 1 else not_favorable)
        return result

    def fit(self, X, y):
        from lale.lib.aif360 import ProtectedAttributesEncoder, Redacting

        fairness_info = {
            "favorable_labels": self.favorable_labels,
            "protected_attributes": self.protected_attributes,
            "unfavorable_labels": self.unfavorable_labels,
        }
        redacting = Redacting(**fairness_info) if self.redact else lale.lib.lale.NoOp
        trainable_redact_and_estim = redacting >> self.estimator
        assert isinstance(trainable_redact_and_estim, lale.operators.TrainablePipeline)
        self.redact_and_estim = trainable_redact_and_estim.fit(X, y)
        self.prot_attr_enc = ProtectedAttributesEncoder(
            **fairness_info,
            remainder="drop",
            return_X_y=True,
        )
        prot_attr_names = [pa["feature"] for pa in self.protected_attributes]
        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label=1,
            unfavorable_label=0,
            protected_attribute_names=prot_attr_names,
        )
        encoded_X, encoded_y = self.prot_attr_enc.transform(X, y)
        self.y_dtype = encoded_y.dtype
        self.y_name = encoded_y.name
        predicted_y = self.redact_and_estim.predict(X)
        predicted_y = _ndarray_to_series(predicted_y, self.y_name, X.index)
        _, predicted_y = self.prot_attr_enc.transform(X, predicted_y)
        predicted_probas = self.redact_and_estim.predict_proba(X)
        dataset_true = self.pandas_to_dataset.convert(encoded_X, encoded_y)
        dataset_pred = self.pandas_to_dataset.convert(
            encoded_X, predicted_y, predicted_probas
        )
        self.mitigator = self.mitigator.fit(dataset_true, dataset_pred)
        self.classes_ = set(list(y))
        self.not_favorable_labels = list(
            self.classes_ - set(list(self.favorable_labels))
        )
        self.classes_ = np.array(list(self.classes_))
        return self

    def predict(self, X):
        predicted_y = self.redact_and_estim.predict(X)
        predicted_probas = self.redact_and_estim.predict_proba(X)
        predicted_y = _ndarray_to_series(predicted_y, self.y_name, X.index)
        encoded_X, predicted_y = self.prot_attr_enc.transform(X, predicted_y)
        dataset_pred = self.pandas_to_dataset.convert(
            encoded_X, predicted_y, predicted_probas
        )
        dataset_out = self.mitigator.predict(dataset_pred)
        _, result_y = dataset_to_pandas(dataset_out, return_only="y")
        decoded_y = self._decode(result_y)
        return decoded_y

    def predict_proba(self, X):
        predicted_y = self.redact_and_estim.predict(X)
        predicted_probas = self.redact_and_estim.predict_proba(X)
        predicted_y = _ndarray_to_series(predicted_y, self.y_name, X.index)
        encoded_X, predicted_y = self.prot_attr_enc.transform(X, predicted_y)
        dataset_pred = self.pandas_to_dataset.convert(
            encoded_X, predicted_y, predicted_probas
        )
        dataset_out = self.mitigator.predict(dataset_pred)
        favorable_probs = dataset_out.scores
        all_probs = np.hstack([1 - favorable_probs, favorable_probs])
        return all_probs


_categorical_supervised_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
    },
}

_categorical_unsupervised_input_fit_schema = {
    "description": "Input data schema for training.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {"description": "Target values; the array is over samples."},
    },
}

_categorical_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_categorical_output_predict_schema = {
    "description": "Predicted class label per sample.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
    ],
}

_categorical_input_predict_proba_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["X"],
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_categorical_output_predict_proba_schema = {
    "description": "The class probabilities of the input samples",
    "anyOf": [
        {"type": "array", "items": {"laleType": "Any"}},
        {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}},
    ],
}

_categorical_input_transform_schema = {
    "description": "Input data schema for transform.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_categorical_output_transform_schema = {
    "description": "Output data schema for reweighted features.",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
    },
}

_numeric_output_transform_schema = {
    "description": "Output data schema for reweighted features.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}


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
