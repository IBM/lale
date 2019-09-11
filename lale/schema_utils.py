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

from typing import Any, Dict, List, Set, Iterable, Iterator, Optional, Tuple, Union, Callable

# Type definitions
Schema = Any
SchemaEnum = Set[Any]

STrue:Schema = {}
SFalse:Schema = {"not":STrue}

def is_true_schema(s:Schema)->bool:
    return s is True or s == STrue

def is_false_schema(s:Schema)->bool:
    return s is False or s == SFalse

def getForOptimizer(obj, prop:str):
    return obj.get(prop + 'ForOptimizer', None)

def getMinimum(obj):
    prop = 'minimum'
    m = obj.get(prop)
    mfo = getForOptimizer(obj, prop)
    if mfo is None:
        return m
    else:
        if m is not None and mfo < m:
            raise ValueError(f"A minimum ({m}) and a *smaller* minimumForOptimizer ({mfo}) was specified in {obj}")
        return mfo

def getMaximum(obj):
    prop = 'maximum'
    m = obj.get(prop)
    mfo = getForOptimizer(obj, prop)
    if mfo is None:
        return m
    else:
        if m is not None and mfo > m:
            raise ValueError(f"A maximum ({m}) and a *greater* maximumForOptimizer ({mfo}) was specified in {obj}")
        return mfo

def getExclusiveMinimum(obj):
    prop = 'exclusveMinimum'
    m = obj.get(prop)
    mfo = getForOptimizer(obj, prop)
    if mfo is None:
        return m
    else:
        return mfo

def getExclusiveMaximum(obj):
    prop = 'exclusiveMaximum'
    m = obj.get(prop)
    mfo = getForOptimizer(obj, prop)
    if mfo is None:
        return m
    else:
        return mfo

def forOptimizer(s:Schema)->bool:
    if isinstance(s, dict):
        return s.get('forOptimizer', True)
    else:
        return True
