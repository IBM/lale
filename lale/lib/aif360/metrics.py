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

import functools
import itertools
from abc import abstractmethod
from typing import Generic, List, Optional, TypeVar, Union

import aif360.metrics
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection

from lale.expressions import count, it
from lale.lib.lale import GroupBy
from lale.lib.rasl import Aggregate
from lale.lib.rasl._metrics import MetricMonoid
from lale.operators import TrainedOperator
from lale.type_checking import JSON_TYPE

from .util import (
    _ensure_str,
    _ndarray_to_series,
    _PandasToDatasetConverter,
    _validate_fairness_info,
    logger,
)

_FAV_LABELS_TYPE = List[Union[int, float, str, List[Union[int, float, str]]]]


class _ScorerFactory:
    _cached_pandas_to_dataset: Optional[_PandasToDatasetConverter]

    def __init__(
        self,
        metric: str,
        favorable_labels: _FAV_LABELS_TYPE,
        protected_attributes: List[JSON_TYPE],
        unfavorable_labels: Optional[_FAV_LABELS_TYPE],
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
        self._cached_pandas_to_dataset = None

    def _pandas_to_dataset(self) -> _PandasToDatasetConverter:
        if self._cached_pandas_to_dataset is None:
            self._cached_pandas_to_dataset = _PandasToDatasetConverter(
                favorable_label=1,
                unfavorable_label=0,
                protected_attribute_names=list(self.privileged_groups[0].keys()),
            )
        return self._cached_pandas_to_dataset

    def _y_pred_series(self, y_true, y_pred, X) -> pd.Series:
        if isinstance(y_pred, pd.Series):
            return y_pred
        assert y_true is not None
        return _ndarray_to_series(
            y_pred,
            y_true.name if isinstance(y_true, pd.Series) else _ensure_str(X.shape[1]),
            X.index if isinstance(X, pd.DataFrame) else None,
            y_pred.dtype,
        )

    def score_data(self, y_true=None, y_pred=None, X=None) -> float:
        assert y_pred is not None
        assert X is not None
        y_pred_orig = y_pred
        y_pred = self._y_pred_series(y_true, y_pred, X)
        encoded_X, y_pred = self.prot_attr_enc.transform(X, y_pred)
        try:
            dataset_pred = self._pandas_to_dataset().convert(encoded_X, y_pred)
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
            dataset_true = self._pandas_to_dataset().convert(encoded_X, y_true)
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

    def score_estimator(self, estimator: TrainedOperator, X, y) -> float:
        return self.score_data(y_true=y, y_pred=estimator.predict(X), X=X)

    def __call__(self, estimator: TrainedOperator, X, y) -> float:
        return self.score_estimator(estimator, X, y)


_Monoid = TypeVar("_Monoid", bound=MetricMonoid)


class _BatchedScorerFactory(_ScorerFactory, Generic[_Monoid]):
    @abstractmethod
    def _to_monoid(self, batch) -> _Monoid:
        pass

    @abstractmethod
    def _from_monoid(self, v) -> float:
        pass

    def score_data_batched(self, batches) -> float:
        lifted_batches = (self._to_monoid(b) for b in batches)
        return self._from_monoid(
            functools.reduce(lambda x, y: x.combine(y), lifted_batches)
        )

    def score_estimator_batched(self, estimator: TrainedOperator, batches) -> float:
        predicted_batches = ((y, estimator.predict(X), X) for X, y in batches)
        return self.score_data_batched(predicted_batches)


class _DIorSPDData(MetricMonoid):
    def __init__(self, priv0_fav0, priv0_fav1, priv1_fav0, priv1_fav1):
        self.priv0_fav0 = priv0_fav0
        self.priv0_fav1 = priv0_fav1
        self.priv1_fav0 = priv1_fav0
        self.priv1_fav1 = priv1_fav1

    def combine(self, other):
        return _DIorSPDData(
            priv0_fav0=self.priv0_fav0 + other.priv0_fav0,
            priv0_fav1=self.priv0_fav1 + other.priv0_fav1,
            priv1_fav0=self.priv1_fav0 + other.priv1_fav0,
            priv1_fav1=self.priv1_fav1 + other.priv1_fav1,
        )


class _DIorSPDScorerFactory(_BatchedScorerFactory[_DIorSPDData]):
    def _to_monoid(self, batch) -> _DIorSPDData:
        if len(batch) == 2:
            X, y_pred = batch
            y_true = None
        else:
            y_true, y_pred, X = batch
        assert y_pred is not None and X is not None, batch
        y_pred = self._y_pred_series(y_true, y_pred, X)
        encoded_X, y_pred = self.prot_attr_enc.transform(X, y_pred)
        df = pd.concat([encoded_X, y_pred], axis=1)
        pa_names = self.privileged_groups[0].keys()
        pipeline = GroupBy(
            by=[it[pa] for pa in pa_names] + [it[y_pred.name]]
        ) >> Aggregate(columns={"count": count(it[y_pred.name])})
        agg_df = pipeline.transform(df)

        def count2(priv, fav):
            row = (priv,) * len(pa_names) + (fav,)
            return agg_df.at[row, "count"] if row in agg_df.index else 0

        return _DIorSPDData(
            priv0_fav0=count2(priv=0, fav=0),
            priv0_fav1=count2(priv=0, fav=1),
            priv1_fav0=count2(priv=1, fav=0),
            priv1_fav1=count2(priv=1, fav=1),
        )


class _AODorEODData(MetricMonoid):
    def __init__(
        self,
        tru0_pred0_priv0,
        tru0_pred0_priv1,
        tru0_pred1_priv0,
        tru0_pred1_priv1,
        tru1_pred0_priv0,
        tru1_pred0_priv1,
        tru1_pred1_priv0,
        tru1_pred1_priv1,
    ):
        self.tru0_pred0_priv0 = tru0_pred0_priv0
        self.tru0_pred0_priv1 = tru0_pred0_priv1
        self.tru0_pred1_priv0 = tru0_pred1_priv0
        self.tru0_pred1_priv1 = tru0_pred1_priv1
        self.tru1_pred0_priv0 = tru1_pred0_priv0
        self.tru1_pred0_priv1 = tru1_pred0_priv1
        self.tru1_pred1_priv0 = tru1_pred1_priv0
        self.tru1_pred1_priv1 = tru1_pred1_priv1

    def combine(self, other):
        return _AODorEODData(
            tru0_pred0_priv0=self.tru0_pred0_priv0 + other.tru0_pred0_priv0,
            tru0_pred0_priv1=self.tru0_pred0_priv1 + other.tru0_pred0_priv1,
            tru0_pred1_priv0=self.tru0_pred1_priv0 + other.tru0_pred1_priv0,
            tru0_pred1_priv1=self.tru0_pred1_priv1 + other.tru0_pred1_priv1,
            tru1_pred0_priv0=self.tru1_pred0_priv0 + other.tru1_pred0_priv0,
            tru1_pred0_priv1=self.tru1_pred0_priv1 + other.tru1_pred0_priv1,
            tru1_pred1_priv0=self.tru1_pred1_priv0 + other.tru1_pred1_priv0,
            tru1_pred1_priv1=self.tru1_pred1_priv1 + other.tru1_pred1_priv1,
        )


class _AODorEODScorerFactory(_BatchedScorerFactory[_AODorEODData]):
    def _to_monoid(self, batch) -> _AODorEODData:
        if len(batch) == 2:
            X, y_pred = batch
            y_true = None
        else:
            y_true, y_pred, X = batch
        assert y_pred is not None and X is not None, batch
        y_pred = self._y_pred_series(y_true, y_pred, X)
        encoded_X, y_pred = self.prot_attr_enc.transform(X, y_pred)

        def is_fresh(col_name):
            assert y_true is not None and isinstance(y_true, pd.Series), batch
            return col_name not in encoded_X.columns and col_name != y_true.name

        if is_fresh("y_pred"):
            y_pred_name = "y_pred"
        else:
            y_pred_name = next(
                f"y_pred_{i}" for i in itertools.count(0) if is_fresh(f"y_pred_{i}")
            )
        y_pred = pd.Series(y_pred, y_pred.index, name=y_pred_name)
        _, y_true = self.prot_attr_enc.transform(X, y_true)
        df = pd.concat([y_true, y_pred, encoded_X], axis=1)
        pa_names = self.privileged_groups[0].keys()
        pipeline = GroupBy(
            by=[it[y_true.name], it[y_pred_name]] + [it[pa] for pa in pa_names]
        ) >> Aggregate(columns={"count": count(it[y_pred.name])})
        agg_df = pipeline.transform(df)

        def count3(tru, pred, priv):
            row = (tru, pred) + (priv,) * len(pa_names)
            return agg_df.at[row, "count"] if row in agg_df.index else 0

        return _AODorEODData(
            tru0_pred0_priv0=count3(tru=0, pred=0, priv=0),
            tru0_pred0_priv1=count3(tru=0, pred=0, priv=1),
            tru0_pred1_priv0=count3(tru=0, pred=1, priv=0),
            tru0_pred1_priv1=count3(tru=0, pred=1, priv=1),
            tru1_pred0_priv0=count3(tru=1, pred=0, priv=0),
            tru1_pred0_priv1=count3(tru=1, pred=0, priv=1),
            tru1_pred1_priv0=count3(tru=1, pred=1, priv=0),
            tru1_pred1_priv1=count3(tru=1, pred=1, priv=1),
        )


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

_BLENDED_SCORER_DOCSTRING = (
    _SCORER_DOCSTRING_ARGS
    + """

    fairness_weight : number, >=0, <=1, default=0.5

      At the default weight of 0.5, the two metrics contribute equally to the blended result. Above 0.5, fairness influences the combination more, and below 0.5, fairness influences the combination less. In the extreme, at 1, the outcome is only determined by fairness, and at 0, the outcome ignores fairness.
"""
    + _SCORER_DOCSTRING_RETURNS
)


class _AccuracyAndDisparateImpact:
    def __init__(
        self,
        favorable_labels: _FAV_LABELS_TYPE,
        protected_attributes: List[JSON_TYPE],
        unfavorable_labels: Optional[_FAV_LABELS_TYPE],
        fairness_weight: float,
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

    def _blend_metrics(self, accuracy: float, symm_di: float) -> float:
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

    def score_data(self, y_true=None, y_pred=None, X=None) -> float:
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        symm_di = self.symm_di_scorer.score_data(y_true, y_pred, X)
        return self._blend_metrics(accuracy, symm_di)

    def score_estimator(self, estimator: TrainedOperator, X, y) -> float:
        accuracy = self.accuracy_scorer(estimator, X, y)
        symm_di = self.symm_di_scorer.score_estimator(estimator, X, y)
        return self._blend_metrics(accuracy, symm_di)

    def __call__(self, estimator: TrainedOperator, X, y) -> float:
        return self.score_estimator(estimator, X, y)


def accuracy_and_disparate_impact(
    favorable_labels: _FAV_LABELS_TYPE,
    protected_attributes: List[JSON_TYPE],
    unfavorable_labels: Optional[_FAV_LABELS_TYPE] = None,
    fairness_weight: float = 0.5,
) -> _AccuracyAndDisparateImpact:
    """
    Create a scikit-learn compatible blended scorer for `accuracy`_
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
    str(accuracy_and_disparate_impact.__doc__) + _BLENDED_SCORER_DOCSTRING
)


class _AverageOddsDifference(_AODorEODScorerFactory):
    def __init__(
        self,
        favorable_labels: _FAV_LABELS_TYPE,
        protected_attributes: List[JSON_TYPE],
        unfavorable_labels: Optional[_FAV_LABELS_TYPE],
    ):
        super().__init__(
            "average_odds_difference",
            favorable_labels,
            protected_attributes,
            unfavorable_labels,
        )

    def _from_monoid(self, v: _AODorEODData) -> float:
        fpr_priv0 = v.tru0_pred1_priv0 / np.float64(
            v.tru0_pred1_priv0 + v.tru0_pred0_priv0
        )
        fpr_priv1 = v.tru0_pred1_priv1 / np.float64(
            v.tru0_pred1_priv1 + v.tru0_pred0_priv1
        )
        tpr_priv0 = v.tru1_pred1_priv0 / np.float64(
            v.tru1_pred1_priv0 + v.tru1_pred0_priv0
        )
        tpr_priv1 = v.tru1_pred1_priv1 / np.float64(
            v.tru1_pred1_priv1 + v.tru1_pred0_priv1
        )
        return 0.5 * (fpr_priv0 - fpr_priv1 + tpr_priv0 - tpr_priv1)


def average_odds_difference(
    favorable_labels: _FAV_LABELS_TYPE,
    protected_attributes: List[JSON_TYPE],
    unfavorable_labels: Optional[_FAV_LABELS_TYPE] = None,
) -> _AverageOddsDifference:
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
    return _AverageOddsDifference(
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
    )


average_odds_difference.__doc__ = (
    str(average_odds_difference.__doc__) + _SCORER_DOCSTRING
)


class _DisparateImpact(_DIorSPDScorerFactory):
    def __init__(
        self,
        favorable_labels: _FAV_LABELS_TYPE,
        protected_attributes: List[JSON_TYPE],
        unfavorable_labels: Optional[_FAV_LABELS_TYPE],
    ):
        super().__init__(
            "disparate_impact",
            favorable_labels,
            protected_attributes,
            unfavorable_labels,
        )

    def _from_monoid(self, v: _DIorSPDData) -> float:
        numerator = v.priv0_fav1 / np.float64(v.priv0_fav0 + v.priv0_fav1)
        denominator = v.priv1_fav1 / np.float64(v.priv1_fav0 + v.priv1_fav1)
        return numerator / denominator


def disparate_impact(
    favorable_labels: _FAV_LABELS_TYPE,
    protected_attributes: List[JSON_TYPE],
    unfavorable_labels: Optional[_FAV_LABELS_TYPE] = None,
) -> _DisparateImpact:
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
    return _DisparateImpact(favorable_labels, protected_attributes, unfavorable_labels)


disparate_impact.__doc__ = str(disparate_impact.__doc__) + _SCORER_DOCSTRING


class _EqualOpportunityDifference(_AODorEODScorerFactory):
    def __init__(
        self,
        favorable_labels: _FAV_LABELS_TYPE,
        protected_attributes: List[JSON_TYPE],
        unfavorable_labels: Optional[_FAV_LABELS_TYPE],
    ):
        super().__init__(
            "equal_opportunity_difference",
            favorable_labels,
            protected_attributes,
            unfavorable_labels,
        )

    def _from_monoid(self, v) -> float:
        tpr_priv0 = v.tru1_pred1_priv0 / np.float64(
            v.tru1_pred1_priv0 + v.tru1_pred0_priv0
        )
        tpr_priv1 = v.tru1_pred1_priv1 / np.float64(
            v.tru1_pred1_priv1 + v.tru1_pred0_priv1
        )
        return tpr_priv0 - tpr_priv1


def equal_opportunity_difference(
    favorable_labels: _FAV_LABELS_TYPE,
    protected_attributes: List[JSON_TYPE],
    unfavorable_labels: Optional[_FAV_LABELS_TYPE] = None,
) -> _EqualOpportunityDifference:
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
    return _EqualOpportunityDifference(
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
        favorable_labels: _FAV_LABELS_TYPE,
        protected_attributes: List[JSON_TYPE],
        unfavorable_labels: Optional[_FAV_LABELS_TYPE],
        fairness_weight: float,
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

    def _blend_metrics(self, r2: float, symm_di: float) -> float:
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

    def score_data(self, y_true=None, y_pred=None, X=None) -> float:
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        symm_di = self.symm_di_scorer.score_data(y_true, y_pred, X)
        return self._blend_metrics(r2, symm_di)

    def score_estimator(self, estimator: TrainedOperator, X, y) -> float:
        r2 = self.r2_scorer(estimator, X, y)
        symm_di = self.symm_di_scorer.score_estimator(estimator, X, y)
        return self._blend_metrics(r2, symm_di)

    def __call__(self, estimator: TrainedOperator, X, y) -> float:
        return self.score_estimator(estimator, X, y)


def r2_and_disparate_impact(
    favorable_labels: _FAV_LABELS_TYPE,
    protected_attributes: List[JSON_TYPE],
    unfavorable_labels: Optional[_FAV_LABELS_TYPE] = None,
    fairness_weight: float = 0.5,
) -> _R2AndDisparateImpact:
    """
    Create a scikit-learn compatible blended scorer for `R2 score`_
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
    str(r2_and_disparate_impact.__doc__) + _BLENDED_SCORER_DOCSTRING
)


class _StatisticalParityDifference(_DIorSPDScorerFactory):
    def __init__(
        self,
        favorable_labels: _FAV_LABELS_TYPE,
        protected_attributes: List[JSON_TYPE],
        unfavorable_labels: Optional[_FAV_LABELS_TYPE],
    ):
        super().__init__(
            "statistical_parity_difference",
            favorable_labels,
            protected_attributes,
            unfavorable_labels,
        )

    def _from_monoid(self, v: _DIorSPDData) -> float:
        minuend = v.priv0_fav1 / np.float64(v.priv0_fav0 + v.priv0_fav1)
        subtrahend = v.priv1_fav1 / np.float64(v.priv1_fav0 + v.priv1_fav1)
        return minuend - subtrahend


def statistical_parity_difference(
    favorable_labels: _FAV_LABELS_TYPE,
    protected_attributes: List[JSON_TYPE],
    unfavorable_labels: Optional[_FAV_LABELS_TYPE] = None,
) -> _StatisticalParityDifference:
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
    return _StatisticalParityDifference(
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
    )


statistical_parity_difference.__doc__ = (
    str(statistical_parity_difference.__doc__) + _SCORER_DOCSTRING
)


class _SymmetricDisparateImpact:
    def __init__(
        self,
        favorable_labels: _FAV_LABELS_TYPE,
        protected_attributes: List[JSON_TYPE],
        unfavorable_labels: Optional[_FAV_LABELS_TYPE],
    ):
        self.disparate_impact_scorer = disparate_impact(
            favorable_labels, protected_attributes, unfavorable_labels
        )

    def _make_symmetric(self, disp_impact: float) -> float:
        if np.isnan(disp_impact):  # empty privileged or unprivileged groups
            return disp_impact
        if disp_impact <= 1.0:
            return disp_impact
        return 1.0 / disp_impact

    def score_data(self, y_true=None, y_pred=None, X=None) -> float:
        disp_impact = self.disparate_impact_scorer.score_data(y_true, y_pred, X)
        return self._make_symmetric(disp_impact)

    def score_estimator(self, estimator: TrainedOperator, X, y) -> float:
        disp_impact = self.disparate_impact_scorer.score_estimator(estimator, X, y)
        return self._make_symmetric(disp_impact)

    def __call__(self, estimator: TrainedOperator, X, y) -> float:
        return self.score_estimator(estimator, X, y)


def symmetric_disparate_impact(
    favorable_labels: _FAV_LABELS_TYPE,
    protected_attributes: List[JSON_TYPE],
    unfavorable_labels: Optional[_FAV_LABELS_TYPE] = None,
) -> _SymmetricDisparateImpact:
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


def theil_index(
    favorable_labels: _FAV_LABELS_TYPE,
    protected_attributes: List[JSON_TYPE],
    unfavorable_labels: Optional[_FAV_LABELS_TYPE] = None,
) -> _ScorerFactory:
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
