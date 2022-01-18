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
from scipy import special

import lale.docstrings
import lale.operators
from lale.expressions import it
from lale.lib.rasl import Map
from lale.lib.sklearn import select_k_best

from ._utils import df_count


# The following functions are a rewriting of sklearn.feature_selection.f_oneway
# Contrary to the sklearn.feature_selection.f_oneway implementation it
# takes as input a dictionary that associate to each class its sample
# and the function is splitted into two parts.
def f_oneway_prep(samples_dict):
    """Performs a 1-way ANOVA.

    Parameters
    ----------
    samples_dict : dictionnary
        The sample measurements associated to each class.

    Returns
    -------
    F-value : float
        The computed F-value of the test.
    p-value : float
        The associated p-value from the F-distribution.
    """
    classes = list(samples_dict.keys())
    n_samples_per_class = {k: a.shape[0] for k, a in samples_dict.items()}
    n_samples = sum(n_samples_per_class.values())
    ss_alldata = sum((a ** 2).sum(axis=0) for a in samples_dict.values())
    sums_samples = {k: np.asarray(a.sum(axis=0)) for k, a in samples_dict.items()}
    sums_alldata = sum(sums_samples.values())
    return (
        classes,
        n_samples_per_class,
        n_samples,
        ss_alldata,
        sums_samples,
        sums_alldata,
    )


def f_oneway_combine(lifted_a, lifted_b):
    (
        classes_a,
        n_samples_per_class_a,
        n_samples_a,
        ss_alldata_a,
        sums_samples_a,
        sums_alldata_a,
    ) = lifted_a
    (
        classes_b,
        n_samples_per_class_b,
        n_samples_b,
        ss_alldata_b,
        sums_samples_b,
        sums_alldata_b,
    ) = lifted_b
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
    return (
        classes,
        n_samples_per_class,
        n_samples,
        ss_alldata,
        sums_samples,
        sums_alldata,
    )


def f_oneway(lifted):
    """Performs a 1-way ANOVA.

    Parameters
    ----------
    samples_dict : dictionnary
        The sample measurements associated to each class.

    Returns
    -------
    F-value : float
        The computed F-value of the test.
    p-value : float
        The associated p-value from the F-distribution.
    """
    (
        classes,
        n_samples_per_class,
        n_samples,
        ss_alldata,
        sums_samples,
        sums_alldata,
    ) = lifted
    n_classes = len(classes)
    square_of_sums_alldata = sums_alldata ** 2
    square_of_sums_args = {k: s ** 2 for k, s in sums_samples.items()}
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


def f_classif_prep(X, y):
    samples_dict = {k: X[y == k] for k in np.unique(y)}
    lifted = f_oneway_prep(samples_dict)
    return lifted


def f_classif(lifted):
    return f_oneway(lifted)


def f_classif_combine(lifted_a, lifted_b):
    return f_oneway_combine(lifted_a, lifted_b)


class _SelectKBestImpl:
    def __init__(
        self,
        score_func_prep=f_classif_prep,
        score_func=f_classif,
        score_func_combine=f_classif_combine,
        *,
        k=10
    ):
        self._hyperparams = {
            "score_func_prep": score_func_prep,
            "score_func": f_classif,
            "score_func_combine": score_func_combine,
            "k": k,
        }
        self.n_samples_seen_ = 0

    def fit(self, X, y=None):
        self._set_fit_attributes(self._lift(X, y, self._hyperparams))
        return self

    def partial_fit(self, X, y=None):
        if self.n_samples_seen_ == 0:  # first fit
            return self.fit(X, y)
        lifted_a = (self.n_samples_seen_, self.feature_names_in_, self.lifted_score_)
        lifted_b = self._lift(X, y, self._hyperparams)
        self._set_fit_attributes(self._combine(lifted_a, lifted_b, self._hyperparams))
        return self

    def transform(self, X):
        if self._transformer is None:
            self._transformer = self._build_transformer()
        return self._transformer.transform(X)

    def _set_fit_attributes(self, lifted):
        self.n_samples_seen_, self.feature_names_in_, self.lifted_score_ = lifted
        score_func = self._hyperparams["score_func"]
        self.scores_, self.pvalues_ = score_func(self.lifted_score_)
        self.n_features_in_ = len(self.feature_names_in_)
        self._transformer = None

    def _build_transformer(self):
        k = self._hyperparams["k"]
        scores = self.scores_.copy()
        scores[np.isnan(scores)] = np.finfo(scores.dtype).min
        ind = np.sort(np.argpartition(scores, -k)[-k:])
        kbest = self.feature_names_in_[ind]
        result = Map(columns={col: it[col] for col in kbest})
        return result

    @staticmethod
    def _lift(X, y, hyperparams):
        score_func_prep = hyperparams["score_func_prep"]
        n_samples_seen = df_count(X)
        feature_names_in = X.columns
        lifted_score = score_func_prep(X, y)
        return n_samples_seen, feature_names_in, lifted_score

    @staticmethod
    def _combine(lifted_a, lifted_b, hyperparams):
        (n_samples_seen_a, feature_names_in_a, lifted_score_a) = lifted_a
        (n_samples_seen_b, feature_names_in_b, lifted_score_b) = lifted_b
        n_samples_seen = n_samples_seen_a + n_samples_seen_b
        assert list(feature_names_in_a) == list(feature_names_in_b)
        feature_names_in = feature_names_in_a
        score_func_combine = hyperparams["score_func_combine"]
        lifted_score = score_func_combine(lifted_score_a, lifted_score_b)
        return n_samples_seen, feature_names_in, lifted_score


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra implementation of SelectKBest.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.select_k_best.html",
    "type": "object",
    "tags": {
        "pre": ["~categoricals"],
        "op": ["transformer", "interpretable"],
        "post": [],
    },
    "properties": {
        "hyperparams": select_k_best._hyperparams_schema,
        "input_fit": select_k_best._input_fit_schema,
        "input_transform": select_k_best._input_transform_schema,
        "output_transform": select_k_best._output_transform_schema,
    },
}

SelectKBest: lale.operators.PlannedIndividualOp
SelectKBest = lale.operators.make_operator(_SelectKBestImpl, _combined_schemas)

lale.docstrings.set_docstrings(SelectKBest)
