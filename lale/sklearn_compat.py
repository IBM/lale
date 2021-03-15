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
import warnings
from typing import Any

# This method (and the to_lale() method on the returned value)
# are the only ones intended to be exported

# This entire file is deprecated, and will be removed soon.
# Please remove all calls to make_sklearn_compat from your code
# as they are no longer needed


def make_sklearn_compat(op):
    """This is a deprecated method for backward compatibility and will be removed soon"""
    warnings.warn(
        "sklearn_compat.make_sklearn_compat exists for backwards compatibility and will be removed soon",
        DeprecationWarning,
    )
    return op


def sklearn_compat_clone(impl: Any) -> Any:
    """This is a deprecated method for backward compatibility and will be removed soon.
    call lale.operators.clone (or scikit-learn clone) instead"""
    warnings.warn(
        "sklearn_compat.sklearn_compat_clone exists for backwards compatibility and will be removed soon",
        DeprecationWarning,
    )
    if impl is None:
        return None

    from sklearn.base import clone

    cp = clone(impl, safe=False)
    return cp
