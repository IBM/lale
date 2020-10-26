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
Schema-enhanced versions of the operators from `autoai_libs`_ to enable hyperparameter tuning.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs

Operators
=========

Preprocessing any columns:

* lale.lib.autoai_libs. `NumpyColumnSelector`_
* lale.lib.autoai_libs. `NumpyReplaceMissingValues`_
* lale.lib.autoai_libs. `float32_transform`_
* lale.lib.autoai_libs. `NumpyPermuteArray`_

Preprocessing categorical columns:

* lale.lib.autoai_libs. `CompressStrings`_
* lale.lib.autoai_libs. `NumpyReplaceUnknownValues`_
* lale.lib.autoai_libs. `boolean2float`_
* lale.lib.autoai_libs. `CatImputer`_
* lale.lib.autoai_libs. `CatEncoder`_

Preprocessing numeric columns:

* lale.lib.autoai_libs. `FloatStr2Float`_
* lale.lib.autoai_libs. `NumImputer`_
* lale.lib.autoai_libs. `OptStandardScaler`_

Feature transformation:

* lale.lib.autoai_libs. `TNoOp`_
* lale.lib.autoai_libs. `TA1`_
* lale.lib.autoai_libs. `TA2`_
* lale.lib.autoai_libs. `TB1`_
* lale.lib.autoai_libs. `TB2`_
* lale.lib.autoai_libs. `TAM`_
* lale.lib.autoai_libs. `TGen`_
* lale.lib.autoai_libs. `FS1`_
* lale.lib.autoai_libs. `FS2`_

.. _`NumpyColumnSelector`: lale.lib.autoai_libs.numpy_column_selector.html
.. _`CompressStrings`: lale.lib.autoai_libs.compress_strings.html
.. _`NumpyReplaceMissingValues`: lale.lib.autoai_libs.numpy_replace_missing_values.html
.. _`NumpyReplaceUnknownValues`: lale.lib.autoai_libs.numpy_replace_unknown_values.html
.. _`boolean2float`: lale.lib.autoai_libs.boolean2float.html
.. _`CatImputer`: lale.lib.autoai_libs.cat_imputer.html
.. _`CatEncoder`: lale.lib.autoai_libs.cat_encoder.html
.. _`float32_transform`: lale.lib.autoai_libs.float32_transform.html
.. _`FloatStr2Float`: lale.lib.autoai_libs.float_str2_float.html
.. _`NumImputer`: lale.lib.autoai_libs.num_imputer.html
.. _`OptStandardScaler`: lale.lib.autoai_libs.opt_standard_scaler.html
.. _`NumpyPermuteArray`: lale.lib.autoai_libs.numpy_permute_array.html
.. _`TNoOp`: lale.lib.autoai_libs.t_no_op.html
.. _`TA1`: lale.lib.autoai_libs.ta1.html
.. _`TA2`: lale.lib.autoai_libs.ta2.html
.. _`TB1`: lale.lib.autoai_libs.tb1.html
.. _`TB2`: lale.lib.autoai_libs.tb2.html
.. _`TAM`: lale.lib.autoai_libs.tam.html
.. _`TGen`: lale.lib.autoai_libs.tgen.html
.. _`FS1`: lale.lib.autoai_libs.fs1.html
.. _`FS2`: lale.lib.autoai_libs.fs2.html
"""

from .boolean2float import boolean2float
from .cat_encoder import CatEncoder
from .cat_imputer import CatImputer
from .compress_strings import CompressStrings
from .float32_transform import float32_transform
from .float_str2_float import FloatStr2Float
from .fs1 import FS1
from .fs2 import FS2
from .num_imputer import NumImputer
from .numpy_column_selector import NumpyColumnSelector
from .numpy_permute_array import NumpyPermuteArray
from .numpy_replace_missing_values import NumpyReplaceMissingValues
from .numpy_replace_unknown_values import NumpyReplaceUnknownValues
from .opt_standard_scaler import OptStandardScaler
from .t_no_op import TNoOp
from .ta1 import TA1
from .ta2 import TA2
from .tam import TAM
from .tb1 import TB1
from .tb2 import TB2
from .tgen import TGen
from .util import wrap_pipeline_segments
