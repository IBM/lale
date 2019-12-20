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

import math
import logging
import numpy

from typing import Any, Dict, List, Set, Iterable, Iterator, Optional, Tuple, Union
from hyperopt import hp
from hyperopt.pyll import scope
from lale.util.VisitorMeta import AbstractVisitorMeta
from lale.search.PGO import FrequencyDistribution

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PGO_input_type = Union[FrequencyDistribution, Iterable[Tuple[Any, int]], None]
class SearchSpace(metaclass=AbstractVisitorMeta):
    def __init__(self, default:Optional[Any]=None):
        self._default = default

    _default:Optional[Any]

    def default(self)->Optional[Any]:
        """Return an optional default value, if None.
            if not None, the default value should be in the
            search space
        """
        return self._default

class SearchSpaceEnum(SearchSpace):
    pgo:Optional[FrequencyDistribution]
    vals:List[Any]
    def __init__(self, vals:Iterable[Any], 
                 pgo:PGO_input_type=None, 
                 default:Optional[Any]=None):
        super(SearchSpaceEnum, self).__init__(default=default)
        self.vals = sorted(vals, key=str)

        if pgo is None or isinstance(pgo, FrequencyDistribution):
            self.pgo = pgo
        else:
            self.pgo = FrequencyDistribution.asEnumValues(pgo, self.vals)

class SearchSpaceConstant(SearchSpaceEnum):
    def __init__(self, v, pgo:PGO_input_type=None):
        super(SearchSpaceConstant, self).__init__([v], pgo=pgo, default=v)

class SearchSpaceBool(SearchSpaceEnum):
    def __init__(self, pgo:PGO_input_type=None, default:Optional[Any]=None):
        super(SearchSpaceBool, self).__init__([True, False], pgo=pgo, default=default)

class SearchSpaceNumber(SearchSpace):
    minimum:Optional[float]
    exclusiveMinumum:bool
    maximum:Optional[float]
    exclusiveMaximum:bool
    discrete:bool
    distribution:str
    pgo:Optional[FrequencyDistribution]

    def __init__(self, 
                 minimum=None, 
                 exclusiveMinimum:bool=False, 
                 maximum=None, 
                 exclusiveMaximum:bool=False, 
                 discrete:bool=False, 
                 distribution="uniform",
                 pgo:PGO_input_type=None,
                 default:Optional[Any]=None) -> None:
        super(SearchSpaceNumber, self).__init__(default=default)
        self.minimum = minimum
        self.exclusiveMinimum = exclusiveMinimum
        self.maximum = maximum
        self.exclusiveMaximum = exclusiveMaximum
        self.distribution = distribution
        self.discrete = discrete
        if pgo is None or isinstance(pgo, FrequencyDistribution):
            self.pgo = pgo
        else:
            if discrete:
                self.pgo = FrequencyDistribution.asIntegerValues(pgo, inclusive_min=self.getInclusiveMin(), inclusive_max=self.getInclusiveMax())
            else:
                self.pgo = FrequencyDistribution.asFloatValues(pgo, inclusive_min=self.getInclusiveMin(), inclusive_max=self.getInclusiveMax())


    def getInclusiveMax(self):
        """ Return the maximum as an inclusive maximum (exclusive maxima are adjusted accordingly)
        """
        max = self.maximum
        if self.exclusiveMaximum:
            if self.discrete:
                max = max - 1
            else:
                max = numpy.nextafter(max, float('-inf'))
        return max

    def getInclusiveMin(self):
        """ Return the maximum as an inclusive minimum (exclusive minima are adjusted accordingly)
        """
        min = self.minimum
        if self.exclusiveMinimum:
            if self.discrete:
                min = min + 1
            else:
                min = numpy.nextafter(min, float('+inf'))
        return min

class SearchSpaceArray(SearchSpace):
    def __init__(self, minimum:int=0, *, maximum:int, contents:SearchSpace, is_tuple=False) -> None:
        super(SearchSpaceArray, self).__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.contents = contents
        self.is_tuple = is_tuple

class SearchSpaceList(SearchSpace):
    def __init__(self, contents:List[SearchSpace], is_tuple=False) -> None:
        super(SearchSpaceList, self).__init__()
        self.contents = contents
        self.is_tuple = is_tuple

class SearchSpaceObject(SearchSpace):
    def __init__(self, longName:str, keys:List[str], choices:Iterable[Any]) -> None:
        super(SearchSpaceObject, self).__init__()
        self.longName = longName
        self.keys = keys
        self.choices = choices
