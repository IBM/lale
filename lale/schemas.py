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

from typing import cast, TypeVar, Any, Dict, List, Tuple, Optional, Union


class Undefined():
    pass


undefined = Undefined()

T = TypeVar('T')
Option = Union[Undefined, T]


class Schema:

    schema: Dict[str, Any]

    def __init__(self,
                 desc: Option[str] = undefined,
                 default: Option[Any] = undefined,
                 forOptimizer: bool = True):
        self.schema: Dict[str, Any] = {}
        if not isinstance(default, Undefined):
            self.schema['default'] = default
        if not isinstance(desc, Undefined):
            self.schema['description'] = desc
        if not forOptimizer:
            self.schema['forOptimizer'] = forOptimizer

    def set(self, prop: str, value: Option[Any]):
        if not isinstance(value, Undefined):
            self.schema[prop] = value


#Base Type

class Bool(Schema):
    def __init__(self,
                 desc: Option[str] = undefined,
                 default: Option[str] = undefined,
                 forOptimizer: bool = True):
        super().__init__(desc, default, forOptimizer)
        self.set('type', 'boolean')


class Enum(Schema):
    def __init__(self,
                 values: List[str] = [],
                 desc: Option[str] = undefined,
                 default: Option[str] = undefined,
                 forOptimizer: bool = True):
        super().__init__(desc, default, forOptimizer)
        self.set('enum', values)


class Float(Schema):
    def __init__(self,
                 desc: Option[str] = undefined,
                 default: Option[str] = undefined,
                 forOptimizer: bool = True,
                 min: Option[float] = undefined,
                 exclusiveMin: Option[bool] = undefined,
                 minForOptimizer: Option[bool] = undefined,
                 max: Option[float] = undefined,
                 exclusiveMax: Option[bool] = undefined,
                 maxForOptimizer: Option[bool] = undefined,
                 distribution: Option[str] = undefined):
        super().__init__(desc, default, forOptimizer)
        self.set('type', 'number')
        self.set('minimum', min)
        self.set('exclusiveMinimum', exclusiveMin)
        self.set('minimumForOptimizer', minForOptimizer)
        self.set('maximum', max)
        self.set('exclusiveMaximum', exclusiveMax)
        self.set('maximumForOptimizer', maxForOptimizer)
        self.set('distribution', distribution)


class Int(Schema):
    def __init__(self,
                 desc: Option[str] = undefined,
                 default: Option[str] = undefined,
                 forOptimizer: bool = True,
                 min: Option[int] = undefined,
                 exclusiveMin: Option[bool] = undefined,
                 max: Option[int] = undefined,
                 exclusiveMax: Option[bool] = undefined,
                 distribution: Option[str] = undefined):
        super().__init__(desc, default, forOptimizer)
        self.set('type', 'integer')
        self.set('minimum', min)
        self.set('exclusiveMinimum', exclusiveMin)
        self.set('maximum', max)
        self.set('exclusiveMaximum', exclusiveMax)
        self.set('distribution', distribution)


class Null(Schema):
    def __init__(self,
                 desc: Option[str] = undefined,
                 forOptimizer: bool = True):
        super().__init__(desc=desc, forOptimizer=forOptimizer)
        self.set('enum', [None])


class Not(Schema):
    def __init__(self,
                 body: Schema):
        super().__init__()
        self.schema = {'not': body.schema}


class JSON(Schema):
    def __init__(self,
                 body: Dict[str, Any]):
        super().__init__()
        self.schema = body

# Combinator


class AnyOf(Schema):
    def __init__(self,
                 types: List[Schema] = [],
                 desc: Option[str] = undefined,
                 default: Option[Any] = undefined):
        super().__init__(desc, default)
        self.set('anyOf', [t.schema for t in types])


class AllOf(Schema):
    def __init__(self,
                 types: List[Schema] = [],
                 desc: Option[str] = undefined,
                 default: Option[Any] = undefined):
        super().__init__(desc, default)
        self.set('allOf', [t.schema for t in types])


class Array(Schema):
    def __init__(self,
                 items: Schema,
                 desc: Option[str] = undefined,
                 default: Option[List[Any]] = undefined,
                 forOptimizer: bool = True,
                 minItems: Option[int] = undefined,
                 minItemsForOptimizer: Option[int] = undefined,
                 maxItems: Option[int] = undefined,
                 maxItemsForOptimizer: Option[int] = undefined,
                 typeForOptimizer: Option[str] = undefined,):
        super().__init__(desc, default, forOptimizer)
        self.set('type', 'array')
        self.set('items', items.schema)
        self.set('minItems', minItems)
        self.set('minItemsForOptimizer', minItemsForOptimizer)
        self.set('maxItems', maxItems)
        self.set('maxItemsForOptimizer', maxItemsForOptimizer)
        self.set('typeForOptimizer', typeForOptimizer)


class Object(Schema):
    def __init__(self,
                 default: Option[Any] = undefined,
                 desc: Option[str] = undefined,
                 forOptimizer: bool = True,
                 required: Option[List[str]] = undefined,
                 additionalProperties: Option[bool] = undefined,
                 **kwargs: Schema):
        super().__init__(desc, default, forOptimizer)
        self.set('type', 'object')
        self.set('required', required)
        self.set('additionalProperties', additionalProperties)
        self.set('properties', {k: p.schema for (k, p) in kwargs.items()})
