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

"""
Scikit-learn compatible wrappers for a subset of the operators from AIF360_ along with schemas to enable hyperparameter tuning.

.. _AIF360: https://github.com/IBM/AIF360

Preprocessing Operators:
========================
* `DisparateImpactRemover`_
* `LFR`_
* `OptimPreproc`_
* `Reweighing`_

Inprocessing Operators:
=======================
* `AdversarialDebiasing`_
* `PrejudiceRemover`_

Postprocessing Operators:
=========================
* `CalibratedEqOddsPostprocessing`_
* `EqOddsPostprocessing`_
* `RejectOptionClassification`_

Metrics:
========
* `accuracy_and_disparate_impact`_
* `disparate_impact`_
* `r2_and_disparate_impact`_
* `statistical_parity_difference`_

Other Operators:
================
* `ProtectedAttributesEncoder`_
* `Redacting`_

Other Functions:
================
* `dataset_fairness_info`_
* `dataset_to_pandas`_

.. _`AdversarialDebiasing`: lale.lib.aif360.adversarial_debiasing.html
.. _`CalibratedEqOddsPostprocessing`: lale.lib.aif360.calibrated_eq_odds_postprocessing.html
.. _`DisparateImpactRemover`: lale.lib.aif360.disparate_impact_remover.html
.. _`EqOddsPostprocessing`: lale.lib.aif360.eq_odds_postprocessing.html
.. _`LFR`: lale.lib.aif360.lfr.html
.. _`OptimPreproc`: lale.lib.aif360.optim_preproc.html
.. _`PrejudiceRemover`: lale.lib.aif360.prejudice_remover.html
.. _`ProtectedAttributesEncoder`: lale.lib.aif360.protected_attributes_encoder.html
.. _`Redacting`: lale.lib.aif360.redacting.html
.. _`RejectOptionClassification`: lale.lib.aif360.reject_option_classification.html
.. _`Reweighing`: lale.lib.aif360.reweighing.html
.. _`accuracy_and_disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.accuracy_and_disparate_impact
.. _`dataset_fairness_info`: lale.lib.aif360.util.html#lale.lib.aif360.util.dataset_fairness_info
.. _`dataset_to_pandas`: lale.lib.aif360.util.html#lale.lib.aif360.util.dataset_to_pandas
.. _`disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.disparate_impact
.. _`r2_and_disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.r2_and_disparate_impact
.. _`statistical_parity_difference`: lale.lib.aif360.util.html#lale.lib.aif360.util.statistical_parity_difference
"""

from .adversarial_debiasing import AdversarialDebiasing
from .calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from .disparate_impact_remover import DisparateImpactRemover
from .eq_odds_postprocessing import EqOddsPostprocessing
from .lfr import LFR
from .optim_preproc import OptimPreproc
from .prejudice_remover import PrejudiceRemover
from .protected_attributes_encoder import ProtectedAttributesEncoder
from .redacting import Redacting
from .reject_option_classification import RejectOptionClassification
from .reweighing import Reweighing
from .util import (
    accuracy_and_disparate_impact,
    dataset_fairness_info,
    dataset_to_pandas,
    disparate_impact,
    fetch_adult_df,
    fetch_creditg_df,
    r2_and_disparate_impact,
    statistical_parity_difference,
)
