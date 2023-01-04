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
* `FairSMOTE`_
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
* `balanced_accuracy_and_disparate_impact`_
* `average_odds_difference`_
* `disparate_impact`_
* `equal_opportunity_difference`_
* `f1_and_disparate_impact`_
* `r2_and_disparate_impact`_
* `statistical_parity_difference`_
* `symmetric_disparate_impact`_
* `theil_index`_

Datasets:
=========
* `fetch_adult_df`_
* `fetch_bank_df`_
* `fetch_compas_df`_
* `fetch_compas_violent_df`_
* `fetch_creditg_df`_
* `fetch_meps_panel19_fy2015_df`_
* `fetch_meps_panel20_fy2015_df`_
* `fetch_meps_panel21_fy2016_df`_
* `fetch_nursery_df`_
* `fetch_ricci_df`_
* `fetch_speeddating_df`_
* `fetch_tae_df`_
* `fetch_titanic_df`_

Other Classes and Operators:
============================
* `FairStratifiedKFold`_
* `ProtectedAttributesEncoder`_
* `Redacting`_

Other Functions:
================
* `count_fairness_groups`_
* `dataset_to_pandas`_
* `fair_stratified_train_test_split`_

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
.. _`FairSMOTE`: lale.lib.aif360.fair_smote.html#lale.lib.aif360.fair_smote.FairSMOTE
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
.. _`balanced_accuracy_and_disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.balanced_accuracy_and_disparate_impact
.. _`count_fairness_groups`: lale.lib.aif360.util.html#lale.lib.aif360.util.count_fairness_groups
.. _`dataset_to_pandas`: lale.lib.aif360.util.html#lale.lib.aif360.util.dataset_to_pandas
.. _`disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.disparate_impact
.. _`equal_opportunity_difference`: lale.lib.aif360.util.html#lale.lib.aif360.util.equal_opportunity_difference
.. _`f1_and_disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.f1_and_disparate_impact
.. _`fair_stratified_train_test_split`: lale.lib.aif360.util.html#lale.lib.aif360.util.fair_stratified_train_test_split
.. _`fetch_adult_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_adult_df
.. _`fetch_bank_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_bank_df
.. _`fetch_compas_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_compas_df
.. _`fetch_compas_violent_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_compas_violent_df
.. _`fetch_creditg_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_creditg_df
.. _`fetch_ricci_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_ricci_df
.. _`fetch_speeddating_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_speeddating_df
.. _`fetch_nursery_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_nursery_df
.. _`fetch_titanic_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_titanic_df
.. _`fetch_meps_panel19_fy2015_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_meps_panel19_fy2015_df
.. _`fetch_meps_panel20_fy2015_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_meps_panel20_fy2015_df
.. _`fetch_meps_panel21_fy2016_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_meps_panel21_fy2016_df
.. _`fetch_tae_df`: lale.lib.aif360.datasets.html#lale.lib.aif360.datasets.fetch_tae_df
.. _`r2_and_disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.r2_and_disparate_impact
.. _`statistical_parity_difference`: lale.lib.aif360.util.html#lale.lib.aif360.util.statistical_parity_difference
.. _`symmetric_disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.symmetric_disparate_impact
.. _`theil_index`: lale.lib.aif360.util.html#lale.lib.aif360.util.theil_index

"""

# Note: all imports should be done as
# from .xxx import XXX as XXX
# this ensures that pyright considers them to be publicly available
# and not private imports (this affects lale users that use pyright)

from .adversarial_debiasing import AdversarialDebiasing as AdversarialDebiasing
from .calibrated_eq_odds_postprocessing import (
    CalibratedEqOddsPostprocessing as CalibratedEqOddsPostprocessing,
)
from .datasets import _fetch_boston_housing_df as _fetch_boston_housing_df
from .datasets import fetch_adult_df as fetch_adult_df
from .datasets import fetch_bank_df as fetch_bank_df
from .datasets import fetch_compas_df as fetch_compas_df
from .datasets import fetch_compas_violent_df as fetch_compas_violent_df
from .datasets import fetch_creditg_df as fetch_creditg_df
from .datasets import fetch_meps_panel19_fy2015_df as fetch_meps_panel19_fy2015_df
from .datasets import fetch_meps_panel20_fy2015_df as fetch_meps_panel20_fy2015_df
from .datasets import fetch_meps_panel21_fy2016_df as fetch_meps_panel21_fy2016_df
from .datasets import fetch_nursery_df as fetch_nursery_df
from .datasets import fetch_ricci_df as fetch_ricci_df
from .datasets import fetch_speeddating_df as fetch_speeddating_df
from .datasets import fetch_tae_df as fetch_tae_df
from .datasets import fetch_titanic_df as fetch_titanic_df
from .disparate_impact_remover import DisparateImpactRemover as DisparateImpactRemover
from .eq_odds_postprocessing import EqOddsPostprocessing as EqOddsPostprocessing
from .fair_smote import FairSMOTE as FairSMOTE
from .gerry_fair_classifier import GerryFairClassifier as GerryFairClassifier
from .lfr import LFR as LFR
from .meta_fair_classifier import MetaFairClassifier as MetaFairClassifier
from .optim_preproc import OptimPreproc as OptimPreproc
from .prejudice_remover import PrejudiceRemover as PrejudiceRemover
from .protected_attributes_encoder import (
    ProtectedAttributesEncoder as ProtectedAttributesEncoder,
)
from .redacting import Redacting as Redacting
from .reject_option_classification import (
    RejectOptionClassification as RejectOptionClassification,
)
from .reweighing import Reweighing as Reweighing
from .util import FAIRNESS_INFO_SCHEMA as FAIRNESS_INFO_SCHEMA
from .util import FairStratifiedKFold as FairStratifiedKFold
from .util import accuracy_and_disparate_impact as accuracy_and_disparate_impact
from .util import average_odds_difference as average_odds_difference
from .util import (
    balanced_accuracy_and_disparate_impact as balanced_accuracy_and_disparate_impact,
)
from .util import count_fairness_groups as count_fairness_groups
from .util import dataset_to_pandas as dataset_to_pandas
from .util import disparate_impact as disparate_impact
from .util import equal_opportunity_difference as equal_opportunity_difference
from .util import f1_and_disparate_impact as f1_and_disparate_impact
from .util import fair_stratified_train_test_split as fair_stratified_train_test_split
from .util import r2_and_disparate_impact as r2_and_disparate_impact
from .util import statistical_parity_difference as statistical_parity_difference
from .util import symmetric_disparate_impact as symmetric_disparate_impact
from .util import theil_index as theil_index
