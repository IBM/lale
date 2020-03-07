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

from .concat_features import ConcatFeatures
from .no_op import NoOp
from .hyperopt import Hyperopt
from .identity_wrapper import IdentityWrapper
from .both import Both
from .sample_based_voting import SampleBasedVoting
from .project import Project
from .batching import Batching
from .grid_search_cv import GridSearchCV
import warnings
with warnings.catch_warnings(record=True) as w:
    try:
        from .smac import SMAC
    except ImportError:
        pass
