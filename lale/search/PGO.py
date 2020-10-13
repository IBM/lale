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

import json
import random
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import jsonschema
import numpy as np

Freqs = Dict[str, int]
PGO = Dict[str, Dict[str, Freqs]]


class DefaultValue(Enum):
    token = 0


_default_value = DefaultValue.token

Def = TypeVar("Def")
Defaultable = Union[DefaultValue, Def]


def remove_defaults(it):
    return filter((lambda x: x[1] is not _default_value), it)


XDK = TypeVar("XDK")
XDV = TypeVar("XDV")


def remove_defaults_dict(d: Dict[XDK, Defaultable[XDV]]) -> Dict[XDK, XDV]:
    return dict(remove_defaults(d.items()))


# utilites to load a pgo from json-ish
def load_pgo_file(filepath) -> PGO:
    with open(filepath) as json_file:
        json_data = json.load(json_file)
        return load_pgo_data(json_data)


def load_pgo_data(json_data) -> PGO:
    jsonschema.validate(json_data, _input_schema, jsonschema.Draft4Validator)
    norm = normalize_pgo_type(json_data)
    return norm


# TODO: Add support for falling back on an underlying distribution
# with some probability
T = TypeVar("T")


class FrequencyDistribution(Generic[T]):
    """ Represents the distribution implied by a histogram
    """

    freq_dist: np.array  # Array[T,int]
    vals: np.array  # Array[T]
    cumulative_freqs: np.array  # Array[int]

    @classmethod
    def asIntegerValues(
        cls,
        freqs: Iterable[Tuple[Any, int]],
        inclusive_min: Optional[int] = None,
        inclusive_max: Optional[int] = None,
    ) -> "FrequencyDistribution[int]":
        freqs = freqsAsIntegerValues(
            freqs, inclusive_min=inclusive_min, inclusive_max=inclusive_max
        )
        return FrequencyDistribution[int](list(freqs), dtype=int)

    @classmethod
    def asFloatValues(
        cls,
        freqs: Iterable[Tuple[Any, int]],
        inclusive_min: Optional[int] = None,
        inclusive_max: Optional[int] = None,
    ) -> "FrequencyDistribution[float]":
        freqs = freqsAsFloatValues(
            freqs, inclusive_min=inclusive_min, inclusive_max=inclusive_max
        )
        return FrequencyDistribution[float](list(freqs), dtype=float)

    @classmethod
    def asEnumValues(
        cls, freqs: Iterable[Tuple[Any, int]], values: List[Any]
    ) -> "FrequencyDistribution[Any]":
        freqs = freqsAsEnumValues(freqs, values=values)
        return FrequencyDistribution[Any](list(freqs), dtype=object)

    def __init__(self, freqs: Iterable[Tuple[Defaultable[T], int]], dtype=object):
        # we need them to be sorted for locality
        sorted_freq_list = sorted(
            freqs,
            key=(
                lambda k: (
                    k[0] is _default_value,
                    None if k[0] is _default_value else k[0],
                )
            ),
        )
        freqs_array = np.array(
            sorted_freq_list, dtype=[("value", object), ("frequency", int)]
        )
        #        freqs_array.sort(order='value')
        self.freq_dist = freqs_array

        self.vals = freqs_array["value"]
        self.cumulative_freqs = np.cumsum(freqs_array["frequency"])

    def __len__(self):
        return np.int_(self.cumulative_freqs[-1])

    @overload
    def __getitem__(self, key: int) -> T:
        ...

    @overload
    def __getitem__(self, key: Sequence[int]) -> Sequence[T]:
        ...

    @overload
    def __getitem__(self, key: slice) -> Sequence[T]:
        ...

    def __getitem__(
        self, key: Union[int, Sequence[int], slice]
    ) -> Union[T, Sequence[T]]:
        indices: Sequence[int]
        single = False
        if isinstance(key, (int, float)):
            single = True
            indices = [key]
        elif isinstance(key, slice):
            # TODO: this could be made more efficient
            indices = range(key.start or 0, key.stop or len(self), key.step or 1)
        else:
            indices = key

        val_indices: Sequence[int] = np.searchsorted(
            self.cumulative_freqs, indices, side="right"
        )

        values = self.vals[val_indices].tolist()

        if single:
            assert len(values) == 1
            return values[0]
        else:
            return values

    def sample(self) -> T:
        ll = len(self)
        i = random.randrange(ll)
        return self[i]

    def samples(self, count: int) -> Sequence[T]:
        ll = len(self)
        i: Sequence[int] = [random.randrange(ll) for _ in range(count)]
        return self[i]


# utiltities to convert and sample from a PGO frequency distribution

DEFAULT_STR = "default"


def freqsAsIntegerValues(
    freqs: Iterable[Tuple[Any, int]],
    inclusive_min: Optional[int] = None,
    inclusive_max: Optional[int] = None,
) -> Iterator[Tuple[Defaultable[int], int]]:
    """ maps the str values to integers, and skips anything that does not look like an integer"""
    for v, f in freqs:
        try:
            if v == DEFAULT_STR:
                yield _default_value, f
                continue
            i = int(v)
            if inclusive_min is not None and inclusive_min > i:
                continue
            if inclusive_max is not None and inclusive_max < i:
                continue
            yield i, f
        except ValueError:
            pass


def freqsAsFloatValues(
    freqs: Iterable[Tuple[Any, int]],
    inclusive_min: Optional[float] = None,
    inclusive_max: Optional[float] = None,
) -> Iterator[Tuple[Defaultable[float], int]]:
    """ maps the str values to integers, and skips anything that does not look like an integer"""
    for v, f in freqs:
        try:
            if v == DEFAULT_STR:
                yield _default_value, f
                continue
            i = float(v)
            if inclusive_min is not None and inclusive_min > i:
                continue
            if inclusive_max is not None and inclusive_max < i:
                continue

            yield i, f
        except ValueError:
            pass


# TODO: we can get a dictionary from freqs (before items() was called)
# and then lookup values in it (since values is likely smaller then freqs)
# or, of course, check which one is smaller and iterate through it
def freqsAsEnumValues(
    freqs: Iterable[Tuple[Any, int]], values: List[Any]
) -> Iterator[Tuple[Defaultable[Any], int]]:
    """ only keeps things that match the string representation of values in the enumeration.
        converts from the string to the value as represented in the enumeration.
    """

    def as_str(v) -> str:
        """ There are some quirks in how the PGO files
        encodes values relative to python's str method
        """
        if v is None:
            return "none"
        elif v is True:
            return "true"
        elif v is False:
            return "false"
        else:
            return str(v)

    value_lookup = {as_str(k): k for k in values}
    for v, f in freqs:
        if v == DEFAULT_STR:
            yield _default_value, f
            continue

        if v in value_lookup:
            yield value_lookup[v], f


_input_type = Dict[str, Dict[str, Union[int, Dict[str, Union[str, int]]]]]


# For now, we skip things of the form
# alg -> {default: number}
# (i.e. without parameters)
def normalize_pgo_type(data: _input_type) -> PGO:
    return {
        alg: {
            param_keys: {
                param_values: int(param_counts)
                for param_values, param_counts in v2.items()
            }
            for param_keys, v2 in v1.items()
            if isinstance(v2, dict)
        }
        for alg, v1 in data.items()
    }


_input_schema: Any = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Input format for pgo files.  Keys are the name of the algorithm",
    "type": "object",
    "additionalProperties": {
        "anyOf": [
            {
                "description": "Keys are the parameter names",
                "type": "object",
                "additionalProperties": {
                    "description": "Keys are value names",
                    "type": "object",
                    "additionalProperties": {
                        "anyOf": [
                            {
                                "description": "the number of times this value was found",
                                "type": "integer",
                            },
                            {
                                "description": "the number of times this value was found",
                                "type": "string",
                            },
                        ]
                    },
                },
            },
            {
                "description": "default value for the optimizer",
                "type": "object",
                "additionalProperties": False,
                "required": ["default"],
                "properties": {
                    "default": {
                        "anyOf": [
                            {
                                "description": "the number of times the default was found",
                                "type": "integer",
                            },
                            {
                                "description": "the number of times the default was found",
                                "type": "string",
                            },
                        ]
                    }
                },
            },
        ]
    },
}
