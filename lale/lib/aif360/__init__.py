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

Operators:
==========
* `CalibratedEqOddsPostprocessing`_
* `DisparateImpactRemover`_
* `EqOddsPostprocessing`_
* `RejectOptionClassification`_

.. _`CalibratedEqOddsPostprocessing`: lale.lib.aif360.calibrated_eq_odds_postprocessing.html
.. _`DisparateImpactRemover`: lale.lib.aif360.disparate_impact_remover.html
.. _`EqOddsPostprocessing`: lale.lib.aif360.eq_odds_postprocessing.html
.. _`RejectOptionClassification`: lale.lib.aif360.reject_option_classification.html

Functions:
==========
* `accuracy_and_disparate_impact`_
* `dataset_fairness_info`_
* `dataset_to_pandas`_
* `disparate_impact`_
* `r2_and_disparate_impact`_
* `statistical_parity_difference`_

.. _`accuracy_and_disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.accuracy_and_disparate_impact
.. _`dataset_fairness_info`: lale.lib.aif360.util.html#lale.lib.aif360.util.dataset_fairness_info
.. _`dataset_to_pandas`: lale.lib.aif360.util.html#lale.lib.aif360.util.dataset_to_pandas
.. _`disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.disparate_impact
.. _`r2_and_disparate_impact`: lale.lib.aif360.util.html#lale.lib.aif360.util.r2_and_disparate_impact
.. _`statistical_parity_difference`: lale.lib.aif360.util.html#lale.lib.aif360.util.statistical_parity_difference
"""

from .calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
from .disparate_impact_remover import DisparateImpactRemover
from .eq_odds_postprocessing import EqOddsPostprocessing
from .reject_option_classification import RejectOptionClassification
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
