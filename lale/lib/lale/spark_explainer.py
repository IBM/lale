# Copyright 2020 IBM Corporation
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
import logging
from typing import Optional, Union

from lale.helpers import _is_spark_df

logger = logging.getLogger(__name__)

# from typing import Literal  # raises a mypy error for <3.8, doesn't for >=3.8
#
# MODE_type = Union[
#     Literal["simple", "extended", "codegen", "cost", "formatted"],
# ]

MODE_type = str


class SparkExplainer:
    def __init__(
        self, extended: Union[bool, MODE_type] = False, mode: Optional[MODE_type] = None
    ):
        self._extended = extended
        self._mode = mode

    def __call__(self, X, y=None):
        if not _is_spark_df(X):
            logger.warning(f"SparkExplain called with non spark data of type {type(X)}")
        else:
            X.explain(extended=self._extended, mode=self._mode)
