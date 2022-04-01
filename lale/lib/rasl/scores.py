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

from typing import Any, Tuple, TypeVar

import numpy as np
from scipy import special
from typing_extensions import Protocol

from lale.expressions import count as agg_count
from lale.expressions import it
from lale.expressions import sum as agg_sum
from lale.helpers import _ensure_pandas, _is_pandas_series
from lale.lib.dataframe import get_columns
from lale.lib.rasl import Aggregate, ConcatFeatures, GroupBy, Map

from .monoid import Monoid, MonoidFactory

ScoreMonoid = Monoid

_InputType = Tuple[Any, Any]  # TODO: be more precise?
_OutputType = Tuple[float, float]
_M = TypeVar("_M", bound=ScoreMonoid)


class ScoreMonoidFactory(MonoidFactory[_InputType, _OutputType, _M], Protocol):
    def score(self, X, y) -> Tuple[float, float]:
        return self.from_monoid(self.to_monoid((X, y)))


class FOnewayData(Monoid):
    def __init__(
        self,
        *,
        classes,
        n_samples_per_class,
        n_samples,
        ss_alldata,
        sums_samples,
        sums_alldata,
    ):
        """
        Parameters
        ----------
        classes: list
            The list of classes.
        n_samples_per_class: dictionary
            The number of samples in each class.
        n_samples: number
            The total number of samples.
        ss_alldata: array
            The sum of square of each feature.
        sums_samples: dictionary
            The sum of each feaure per class.
        sums_alldata: array
            The sum of each feaure.
        """
        self.classes = classes
        self.n_samples_per_class = n_samples_per_class
        self.n_samples = n_samples
        self.ss_alldata = ss_alldata
        self.sums_samples = sums_samples
        self.sums_alldata = sums_alldata

    def combine(self, other: "FOnewayData"):
        classes_a = self.classes
        n_samples_per_class_a = self.n_samples_per_class
        n_samples_a = self.n_samples
        ss_alldata_a = self.ss_alldata
        sums_samples_a = self.sums_samples
        sums_alldata_a = self.sums_alldata
        classes_b = other.classes
        n_samples_per_class_b = other.n_samples_per_class
        n_samples_b = other.n_samples
        ss_alldata_b = other.ss_alldata
        sums_samples_b = other.sums_samples
        sums_alldata_b = other.sums_alldata
        classes = list(set(classes_a + classes_b))
        n_samples_per_class = {
            k: (n_samples_per_class_a[k] if k in n_samples_per_class_a else 0)
            + (n_samples_per_class_b[k] if k in n_samples_per_class_b else 0)
            for k in classes
        }
        n_samples = n_samples_a + n_samples_b
        ss_alldata = ss_alldata_a + ss_alldata_b
        sums_samples = {
            k: (sums_samples_a[k] if k in sums_samples_a else 0)
            + (sums_samples_b[k] if k in sums_samples_b else 0)
            for k in classes
        }
        sums_alldata = sums_alldata_a + sums_alldata_b
        return FOnewayData(
            classes=classes,
            n_samples_per_class=n_samples_per_class,
            n_samples=n_samples,
            ss_alldata=ss_alldata,
            sums_samples=sums_samples,
            sums_alldata=sums_alldata,
        )


def _gen_name(base, avoid):
    if base not in avoid:
        return base
    cpt = 0
    while f"{base}{cpt}" in avoid:
        cpt += 1
    return f"{base}{cpt}"


# The following function is a rewriting of sklearn.feature_selection.f_oneway
# Compared to the sklearn.feature_selection.f_oneway implementation it
# takes as input the dataset and the target vector.
# Moreover, the function is splitted into two parts: `_f_oneway_lift` and
# `_f_oneway_lower`.
def _f_oneway_lift(X, y) -> FOnewayData:
    """Prepare the data for a 1-way ANOVA.

    Parameters
    ----------
    X: array
        The sample measurements.
    y: array
        The target vector.

    Returns
    -------
    monoid: FOnewayData
        The inermediate data that can be combine for incremental computation.
    """
    if get_columns(y)[0] is None:
        if _is_pandas_series(y):
            y = y.rename(_gen_name("target", get_columns(X)))
    Xy = ConcatFeatures().transform([X, y])
    X_by_y = GroupBy(by=[it[get_columns(y)[0]]]).transform(Xy)

    agg_sum_cols = Aggregate(columns={col: agg_sum(it[col]) for col in get_columns(X)})
    sums_samples = _ensure_pandas(agg_sum_cols.transform(X_by_y))
    n_samples_per_class = Aggregate(
        columns={"n_samples_per_class": agg_count(it[get_columns(X)[0]])}
    ).transform(X_by_y)
    n_samples = _ensure_pandas(
        Aggregate(columns={"sum": agg_sum(it["n_samples_per_class"])}).transform(
            n_samples_per_class
        )
    )["sum"][0]
    sqr_cols = Map(columns={col: it[col] ** 2 for col in get_columns(X)})
    ss_alldata = _ensure_pandas((sqr_cols >> agg_sum_cols).transform(X)).loc[0]
    sums_alldata = _ensure_pandas(agg_sum_cols.transform(X)).loc[0].to_numpy()
    n_samples_per_class = _ensure_pandas(n_samples_per_class).to_dict()[
        "n_samples_per_class"
    ]
    classes = list(n_samples_per_class.keys())
    sums_samples = {k: sums_samples.loc[k].to_numpy() for k in classes}

    return FOnewayData(
        classes=classes,
        n_samples_per_class=n_samples_per_class,
        n_samples=n_samples,
        ss_alldata=ss_alldata,
        sums_samples=sums_samples,
        sums_alldata=sums_alldata,
    )


def _f_oneway_lower(lifted: FOnewayData):
    """Performs a 1-way ANOVA.

    Parameters
    ----------
    lifted : FOnewayData
        The result of `to_monoid`.

    Returns
    -------
    F-value : float
        The computed F-value of the test.
    p-value : float
        The associated p-value from the F-distribution.
    """
    classes = lifted.classes
    n_samples_per_class = lifted.n_samples_per_class
    n_samples = lifted.n_samples
    ss_alldata = lifted.ss_alldata
    sums_samples = lifted.sums_samples
    sums_alldata = lifted.sums_alldata
    n_classes = len(classes)
    square_of_sums_alldata = sums_alldata**2
    square_of_sums_args = {k: s**2 for k, s in sums_samples.items()}
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    ssbn = 0.0
    for k in n_samples_per_class:
        ssbn += square_of_sums_args[k] / n_samples_per_class[k]
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    # constant_features_idx = np.where(msw == 0.0)[0]
    # if (np.nonzero(msb)[0].size != msb.size and constant_features_idx.size):
    #     warnings.warn("Features %s are constant." % constant_features_idx,
    #                   UserWarning)
    f = msb / msw
    # flatten matrix to vector in sparse case
    f = np.asarray(f).ravel()
    prob = special.fdtrc(dfbn, dfwn, f)
    return f, prob


class FClassif(ScoreMonoidFactory[FOnewayData]):
    """Compute the ANOVA F-value for the provided sample."""

    def to_monoid(self, v: Tuple[Any, Any]) -> FOnewayData:
        X, y = v
        return _f_oneway_lift(X, y)

    def from_monoid(self, lifted: FOnewayData):
        return _f_oneway_lower(lifted)
