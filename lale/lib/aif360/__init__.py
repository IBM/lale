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

"""Scikit-learn compatible wrappers for several operators and metrics from AIF360_ along with schemas to enable hyperparameter tuning.

.. _AIF360: https://github.com/IBM/AIF360

All operators and metrics in the Lale wrappers for AIF360 take two
arguments, `favorable_labels` and `protected_attributes`, collectively
referred to as *fairness info*. For example, the following code
indicates that the reference group comprises male values in the
`personal_status` attribute as well as values from 26 to 1000 in the
`age` attribute.

.. code:: Python

    creditg_fairness_info = {
        "favorable_labels": ["good"],
        "protected_attributes": [
            {
                "feature": "personal_status",
                "reference_group": [
                    "male div/sep", "male mar/wid", "male single",
                ],
            },
            {"feature": "age", "reference_group": [[26, 1000]]},
        ],
    }

See the following notebooks for more detailed examples:

* https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/demo_aif360.ipynb
* https://nbviewer.jupyter.org/github/IBM/watson-machine-learning-samples/blob/master/cloud/notebooks/python_sdk/experiments/autoai/Use%20Lale%20AIF360%20scorers%20to%20calculate%20and%20mitigate%20bias%20for%20credit%20risk%20AutoAI%20model.ipynb

Pre-Estimator Mitigation Operators:
===================================
* `DisparateImpactRemover`_
* `LFR`_
* `Reweighing`_

In-Estimator Mitigation Operators:
==================================
* `AdversarialDebiasing`_
* `GerryFairClassifier`_
* `MetaFairClassifier`_
* `PrejudiceRemover`_

Post-Estimator Mitigation Operators:
====================================
* `CalibratedEqOddsPostprocessing`_
* `EqOddsPostprocessing`_
* `RejectOptionClassification`_

Metrics:
========
* `accuracy_and_disparate_impact`_
* `average_odds_difference`_
* `disparate_impact`_
* `equal_opportunity_difference`_
* `r2_and_disparate_impact`_
* `statistical_parity_difference`_
* `theil_index`_

Other Classes and Operators:
============================
* `FairStratifiedKFold`_
* `ProtectedAttributesEncoder`_
* `Redacting`_

Other Functions:
================
* `dataset_to_pandas`_
* `fair_stratified_train_test_split`_
* `fetch_adult_df`_
* `fetch_bank_df`_
* `fetch_compas_df`_
* `fetch_creditg_df`_
* `fetch_ricci_df`_

Mitigator Patterns:
===================

AIF360 provides three kinds of fairness mitigators, illustrated in the
following picture. *Pre-estimator* mitigators transform the data
before it gets to an estimator; *in-estimator* mitigators include
their own estimator; and *post-estimator* mitigators transform
predictions after those come back from an estimator.

.. image:: ../../docs/img/fairness_patterns.png

In the picture, italics indicate parameters of the pattern.
For example, consider the following code:

.. code:: Python

    pipeline = LFR(
        **fairness_info,
        preparation=(
            (Project(columns={"type": "string"}) >> OneHotEncoder(handle_unknown="ignore"))
            & Project(columns={"type": "number"})
        )
        >> ConcatFeatures
    ) >> LogisticRegression(max_iter=1000)

In this example, the *mitigator* is LFR (which is pre-estimator), the
*estimator* is LogisticRegression, and the *preparation* is a
sub-pipeline that one-hot-encodes strings. If all features of the data
are numerical, then the preparation can be omitted. Internally, the
LFR higher-order operator uses two auxiliary operators, Redacting
and ProtectedAttributesEncoder.  Redacting sets protected attributes
to a constant to prevent them from directly influencing
fairness-agnostic data preparation or estimators. And the
ProtectedAttributesEncoder encodes protected attributes and labels as
zero or one to simplify the task for the mitigator.


.. _`AdversarialDebiasing`: lale.lib.aif360.adversarial_debiasing.html#lale.lib.aif360.adversarial_debiasing.AdversarialDebiasing
.. _`CalibratedEqOddsPostprocessing`: lale.lib.aif360.calibrated_eq_odds_postprocessing.html#lale.lib.aif360.calibrated_eq_odds_postprocessing.CalibratedEqOddsPostprocessing
.. _`DisparateImpactRemover`: lale.lib.aif360.disparate_impact_remover.html#lale.lib.aif360.disparate_impact_remover.DisparateImpactRemover
.. _`EqOddsPostprocessing`: lale.lib.aif360.eq_odds_postprocessing.html#lale.lib.aif360.eq_odds_postprocessing.EqOddsPostprocessing
.. _`FairStratifiedKFold`: lale.lib.aif360.util.html#lale.lib.aif360.util.FairStratifiedKFold
.. _`LFR`: lale.lib.aif360.lfr.html#lale.lib.aif360.lfr.LFR
.. _`GerryFairClassifier`: lale.lib.aif360.gerry_fair_classifier.html#lale.lib.aif360.gerry_fair_classifier.GerryFairClassifier
.. _`MetaFairClassifier`: lale.lib.aif360.meta_fair_classifier.html#lale.lib.aif360.meta_fair_classifier.MetaFairClassifier
.. _`OptimPreproc`: lale.lib.aif360.optim_preproc.html#lale.lib.aif360.optim_preproc.OptimPreproc
.. _`PrejudiceRemover`: lale.lib.aif360.prejudice_remover.html#lale.lib.aif360.prejudice_remover.PrejudiceRemover
.. _`ProtectedAttributesEncoder`: lale.lib.aif360.protected_attributes_encoder.html#lale.lib.aif360.protected_attributes_encoder.ProtectedAttributesEncoder
.. _`Redacting`: lale.lib.aif360.redacting.html#lale.lib.aif360.redacting.Redacting
.. _`RejectOptionClassification`: lale.lib.aif360.reject_option_classification.html#lale.lib.aif360.reject_option_classification.RejectOptionClassification
.. _`Reweighing`: lale.lib.aif360.reweighing.html#lale.lib.aif360.reweighing.Reweighing
.. _`accuracy_and_disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.accuracy_and_disparate_impact
.. _`average_odds_difference`: lale.lib.aif360.util.html#lale.lib.aif360.util.average_odds_difference
.. _`dataset_to_pandas`: lale.lib.aif360.util.html#lale.lib.aif360.util.dataset_to_pandas
.. _`disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.disparate_impact
.. _`equal_opportunity_difference`: lale.lib.aif360.util.html#lale.lib.aif360.util.equal_opportunity_difference
.. _`fair_stratified_train_test_split`: lale.lib.aif360.util.html#lale.lib.aif360.util.fair_stratified_train_test_split
.. _`fetch_adult_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_adult_df
.. _`fetch_bank_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_bank_df
.. _`fetch_compas_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_compas_df
.. _`fetch_creditg_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_creditg_df
.. _`fetch_ricci_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_ricci_df
.. _`r2_and_disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.r2_and_disparate_impact
.. _`statistical_parity_difference`: lale.lib.aif360.util.html#lale.lib.aif360.util.statistical_parity_difference
.. _`theil_index`: lale.lib.aif360.util.html#lale.lib.aif360.util.theil_index

"""

from .adversarial_debiasing import AdversarialDebiasing
from .calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from .datasets import (
    fetch_adult_df,
    fetch_bank_df,
    fetch_compas_df,
    fetch_creditg_df,
    fetch_ricci_df,
)
from .disparate_impact_remover import DisparateImpactRemover
from .eq_odds_postprocessing import EqOddsPostprocessing
from .gerry_fair_classifier import GerryFairClassifier
from .lfr import LFR
from .meta_fair_classifier import MetaFairClassifier
from .optim_preproc import OptimPreproc
from .prejudice_remover import PrejudiceRemover
from .protected_attributes_encoder import ProtectedAttributesEncoder
from .redacting import Redacting
from .reject_option_classification import RejectOptionClassification
from .reweighing import Reweighing
from .util import (
    FAIRNESS_INFO_SCHEMA,
    FairStratifiedKFold,
    accuracy_and_disparate_impact,
    average_odds_difference,
    dataset_to_pandas,
    disparate_impact,
    equal_opportunity_difference,
    fair_stratified_train_test_split,
    r2_and_disparate_impact,
    statistical_parity_difference,
    theil_index,
)
