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
import os

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

class SearchSpaceEmpty(SearchSpace):
    def __init__(self):
        super(SearchSpaceEmpty, self).__init__()

    def __str__(sefl):
        return "***EMPTY***"

class SearchSpacePrimitive(SearchSpace):
    def __init__(self, default:Optional[Any]=None):
        super(SearchSpacePrimitive, self).__init__(default=default)

class SearchSpaceEnum(SearchSpacePrimitive):
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

    def __str__(self):
        return "<" + ",".join(map(str, self.vals)) + ">"

class SearchSpaceConstant(SearchSpaceEnum):
    def __init__(self, v, pgo:PGO_input_type=None):
        super(SearchSpaceConstant, self).__init__([v], pgo=pgo, default=v)

    def __str__(self):
        return str(self.vals[0])

class SearchSpaceBool(SearchSpaceEnum):
    def __init__(self, pgo:PGO_input_type=None, default:Optional[Any]=None):
        super(SearchSpaceBool, self).__init__([True, False], pgo=pgo, default=default)

class SearchSpaceNumber(SearchSpacePrimitive):
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

    def __str__(self):
        ret:str = ""
        if self.exclusiveMinimum or self.minimum is None:
             ret += "("
        else:

            ret += "["
        if self.discrete:
            ret += '\u2308'

        if self.minimum is None:
            ret += '\u221E'
        else:
            ret += str(self.minimum)

        if not self.distribution or self.distribution == "uniform" or self.distribution == "integer":
            ret += ","
        elif self.distribution == "loguniform":
            ret += ",<log>,"
        else:
            ret += ",<" + self.distribution + ">,"

        if self.maximum is None:
            ret += '\u221E'
        else:
            ret += str(self.maximum)

        if self.discrete:
            ret += '\u2309'
        if self.exclusiveMaximum or self.maximum is None:
             ret += ")"
        else:
            ret += "]"
        return ret

class SearchSpaceArray(SearchSpace):
    def __init__(self, prefix:Optional[List[SearchSpace]], minimum:int=0, *, maximum:int, additional:Optional[SearchSpace]=None, is_tuple=False) -> None:
        super(SearchSpaceArray, self).__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.prefix = prefix
        self.additional = additional
        self.is_tuple = is_tuple
    
    def __str__(self):
        ret:str = ""
        ret += f"Array<{self.minimum}, {self.maximum}>"
        if self.is_tuple:
            ret += "("
        else:
            ret += "["

        if self.prefix is not None:
            ret += ",".join(map(str,self.prefix))
            if self.additional is not None:
                ret += ","
        if self.additional is not None:
            ret += "...,"
            ret += str(self.additional)

        if self.is_tuple:
            ret += ")"
        else:
            ret += "]"
        return ret

    def items(self, max:Optional[int]=None)->Iterable[SearchSpace]:
        prefix_len:int
        if self.prefix is not None:
            prefix_len = len(self.prefix)
        else:
            prefix_len = 0

        num_elts = self.maximum
        if max is not None:
            num_elts = min(num_elts, max)

        for i in range(num_elts):
            if self.prefix is not None and i < prefix_len:
                yield self.prefix[i]
            else:
                if self.additional is not None:
                    yield self.additional

class SearchSpaceObject(SearchSpace):
    def __init__(self, longName:str, keys:List[str], choices:Iterable[Any]) -> None:
        super(SearchSpaceObject, self).__init__()
        self.longName = longName
        self.keys = keys
        self.choices = choices

    def __str__(self):
        ret:str = ""
        ret += f"Object<{self.longName}>["

        choice_strs:List[str] = []
        for c in self.choices:
            opts:List[str] = []
            for k,v in zip(self.keys, c):
                opts.append(k + "->" + str(v))
            l = ";".join(opts)
            choice_strs.append("{" + l + "}")

        ret += ",".join(choice_strs) + "]"

        return ret

class SearchSpaceSum(SearchSpace):
    sub_spaces:List[SearchSpace]
    def __init__(self, 
                 sub_spaces:List[SearchSpace], 
                 default:Optional[Any]=None):
        super(SearchSpaceSum, self).__init__(default=default)
        self.sub_spaces = sub_spaces

    def __str__(self):
        ret:str = "\u2211["
        ret += "|".join(map(str, self.sub_spaces))
        ret += "]"
        return ret

class SearchSpaceOperator(SearchSpace):
    sub_space:SearchSpace
    def __init__(self,
                 sub_space:SearchSpace,
                 default:Optional[Any]=None):
        super(SearchSpaceOperator, self).__init__(default=default)
        self.sub_space = sub_space

    def __str__(self):
        ret:str = "\u00AB"
        ret += str(self.sub_space)
        ret += "\u00BB"
        return ret

class SearchSpaceProduct(SearchSpace):
    sub_spaces:List[Tuple[str, SearchSpace]]
    def __init__(self, 
                 sub_spaces:List[Tuple[str,SearchSpace]],
                 default:Optional[Any]=None):
        super(SearchSpaceProduct, self).__init__(default=default)
        self.sub_spaces = sub_spaces
    
    def get_indexed_spaces(self)->Iterable[Tuple[str, int, SearchSpace]]:
        indices:Dict[str,int]={}
        def make_indexed(name:str)->Tuple[str,int]:
            idx = 0
            if name in indices:
                idx = indices[name] + 1
                indices[name] = idx
            else:
                indices[name] = 0
            return (name, idx)
        def enhance_tuple(x:Tuple[str,int], space:SearchSpace)->Tuple[str,int,SearchSpace]:
            return (x[0], x[1], space)
        
        return [enhance_tuple(make_indexed(name), space) for name,space in self.sub_spaces]

    def __str__(self):
        ret:str = "\u220F{"

        parts:List[str] = []
        for k,v in self.sub_spaces:
            parts.append(k + "->" + str(v))
        ret = ";".join(parts)

        ret += "}"
        return ret

# for debugging
_print_search_space_env_options:Optional[Set[str]] = None

def _get_print_search_space_options()->Set[str]:
    global _print_search_space_env_options
    options:Set[str]
    if _print_search_space_env_options is None:
        debug = os.environ.get("LALE_PRINT_SEARCH_SPACE", None)
        if debug is None:
            options = set()
        else:
            options_raw = debug.split(",")
            options = set(s.strip().lower() for s in options_raw)
        _print_search_space_env_options = options
    else:
        options = _print_search_space_env_options
    return options

def should_print_search_space(*s:str):
    options:Set[str] = _get_print_search_space_options()
    for x in s:
        if x in options:
            return True
    return False
  