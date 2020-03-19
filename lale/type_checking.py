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

"""Lale uses `JSON Schema`_ to check machine-learning pipelines for correct types.

In general, there are two kinds of checks. The first is an instance
check (`v: s`), which checks whether a JSON value v is valid for a
schema s. The second is a subschema_ check (`s <: t`), which checks
whether one schema s is a subchema of another schema t.

Besides regular JSON values, Lale also supports certain JSON-like
values. For example, a ``np.ndarray`` of numbers is treated like a
JSON array of arrays of numbers. Furthermore, Lale supports an 'Any'
type for which all instance and subschema checks on the left as well
as the right side succeed. This is specified using ``{'laleType': 'Any'}``.

.. _`JSON Schema`: https://json-schema.org/understanding-json-schema/reference/

.. _subschema: https://arxiv.org/abs/1911.12651
"""

import functools
import jsonschema
import jsonsubschema
import lale.helpers
import numpy as np
import pandas as pd
import scipy.sparse
import logging
import inspect
from typing import Any, Dict, List, Union
import lale.datasets.data_schemas

def validate_schema(value, schema: Dict[str, Any], subsample_array:bool=True):
    """Validate that the value is an instance of the schema.

    Parameters
    ----------
    value: JSON (int, float, str, list, dict) or JSON-like (tuple, np.ndarray, pd.DataFrame ...).
        Left-hand side of instance check.

    schema: JSON schema
        Right-hand side of instance check.

    subsample_array: bool
        Speed up checking by doing only partial conversion to JSON.

    Raises
    ------
    jsonschema.ValidationError
        The value was invalid for the schema.
    """
    json_value = lale.helpers.data_to_json(value, subsample_array)
    jsonschema.validate(json_value, schema, jsonschema.Draft4Validator)

def _json_replace(subject, old, new):
    if subject == old:
        return new
    if isinstance(subject, list):
        result = [_json_replace(s, old, new) for s in subject]
        for i in range(len(subject)):
            if subject[i] != result[i]:
                return result
    elif isinstance(subject, tuple):
        result = tuple([_json_replace(s, old, new) for s in subject])
        for i in range(len(subject)):
            if subject[i] != result[i]:
                return result
    elif isinstance(subject, dict):
        if isinstance(old, dict):
            is_sub_dict = True
            for k, v in old.items():
                if k not in subject or subject[k] != v:
                    is_sub_dict = False
                    break
            if is_sub_dict:
                return {**subject, **new}
        result = {k: _json_replace(v, old, new) for k, v in subject.items()}
        for k in subject:
            if subject[k] != result[k]:
                return result
    return subject #nothing changed so share original object (not a copy)

def is_subschema(sub_schema, super_schema) -> bool:
    """Is sub_schema a subschema of super_schema?

    Parameters
    ----------
    sub_schema: JSON schema
        Left-hand side of subschema check.

    super_schema: JSON schema
        Right-hand side of subschema check.

    Returns
    -------
    bool
        True if `sub_schema <: super_schema`, False otherwise.
    """
    new_sub = _json_replace(sub_schema, {'laleType': 'Any'}, {'not': {}})
    try:
        return jsonsubschema.isSubschema(new_sub, super_schema)
    except Exception as e:
        raise ValueError(f'problem checking ({sub_schema} <: {super_schema})') from e

class SubschemaError(Exception):
    """Raised when a subschema check (sub `<:` sup) failed.
    """
    def __init__(self, sub, sup, sub_name='sub', sup_name='super'):
        self.sub = sub
        self.sup = sup
        self.sub_name = sub_name
        self.sup_name = sup_name

    def __str__(self):
        summary = f'Expected {self.sub_name} to be a subschema of {self.sup_name}.'
        import lale.pretty_print
        sub = lale.pretty_print.schema_to_string(self.sub)
        sup = lale.pretty_print.schema_to_string(self.sup)
        details = f'\n{self.sub_name} = {sub}\n{self.sup_name} = {sup}'
        return summary + details

def _validate_subschema(sub, sup, sub_name='sub', sup_name='super'):
    if not is_subschema(sub, sup):
        raise SubschemaError(sub, sup, sub_name, sup_name)

def validate_schema_or_subschema(lhs, super_schema):
    """Validate that lhs is an instance of or a subschema of super_schema.

    Parameters
    ----------
    lhs: value or JSON schema
        Left-hand side of instance or subschema check.

    super_schema: JSON schema
        Right-hand side of instance or subschema check.

    Raises
    ------
    jsonschema.ValidationError
        The lhs was an invalid value for super_schema.

    SubschemaError
        The lhs was or had a schema that was not a subschema of super_schema.
    """
    if lale.helpers.is_schema(lhs):
        sub_schema = lhs
    else:
        try:
            sub_schema = lale.datasets.data_schemas.to_schema(lhs)
        except ValueError as e:
            sub_schema = None
    if sub_schema is None:
        validate_schema(lhs, super_schema)
    else:
        _validate_subschema(sub_schema, super_schema)

def join_schemas(*schemas: list):
    """Compute the lattice join (union type, disjunction) of the arguments.

    Parameters
    ----------
    *schemas: list of JSON schemas
        Schemas to be joined.

    Returns
    -------
    JSON schema
        The joined schema.
    """
    def join_two_schemas(s_a, s_b):
        if s_a is None:
            return s_b
        s_a = lale.helpers.dict_without(s_a, 'description')
        s_b = lale.helpers.dict_without(s_b, 'description')
        if is_subschema(s_a, s_b):
            return s_b
        if is_subschema(s_b, s_a):
            return s_a
        return jsonsubschema.joinSchemas(s_a, s_b)
    if len(schemas) == 0:
        return {'not':{}}
    return functools.reduce(join_two_schemas, schemas)
        
def get_hyperparam_names(op: 'lale.operators.IndividualOp') -> List[str]:
    """Names of the arguments to the constructor of the impl.

    Parameters
    ----------
    op: lale.operators.IndividualOp
        Operator whose hyperparameters to get.

    Returns
    -------
    List[str]
        List of hyperparameter names.
    """
    if op._impl.__module__.startswith('lale'):
        hp_schema = op.hyperparam_schema()
        params = next(iter(hp_schema.get('allOf', []))).get('properties', {})
        return list(params.keys())
    else:
        return inspect.getargspec(op._impl_class().__init__).args
                
def validate_method(op: 'lale.operators.IndividualOp', schema_name: str):
    """Check whether the operator has the given method schema.

    Parameters
    ----------
    op: lale.operators.IndividualOp
        Operator whose methods to check.

    schema_name: 'input_fit' or 'input_predict' or 'input_predict_proba' or 'input_transform' 'output_predict' or 'output_predict_proba' or 'output_transform'
        Name of schema to check.

    Raises
    ------
    AssertionError
        The operator does not have the given schema.
    """
    if op._impl.__module__.startswith('lale'):
        assert schema_name in op._schemas['properties']
    else:
        method_name = ''
        if schema_name.startswith('input_'):
            method_name = schema_name[len('input_'):]
        elif schema_name.startswith('output_'):
            method_name = schema_name[len('output_'):]
        if method_name:
            assert hasattr(op._impl, method_name)

def _get_args_schema(fun):
    sig = inspect.signature(fun)
    result = {'type': 'object', 'properties': {}}
    required = []
    additional_properties = False
    for name, param in sig.parameters.items():
        ignored_kinds = [inspect.Parameter.VAR_POSITIONAL,
                         inspect.Parameter.VAR_KEYWORD]
        if name != 'self':
            if param.kind in ignored_kinds:
                additional_properties = True
            else:
                if param.default == inspect.Parameter.empty:
                    param_schema = {'laleType': 'Any'}
                    required.append(name)
                else:
                    param_schema = {'default': param.default}
                result['properties'][name] = param_schema
    if not additional_properties:
        result['additionalProperties'] = False
    if len(required) > 0:
        result['required'] = required
    return result

def get_default_schema(impl):
    """Creates combined schemas for a bare operator implementation class.

    Used when there were no explicit combined schemas provided when
    the operator was created. The default schema provides defaults by
    inspecting the signature of the ``__init__`` method, and uses
    'Any' types for the inputs and outputs of other methods.

    Returns
    -------
    JSON Schema
        Combined schema with properties for hyperparams and
        all applicable method inputs and outputs.
    """
    if hasattr(impl, '__init__'):
        hyperparams_schema = _get_args_schema(impl.__init__)
    else:
        hyperparams_schema = {'type': 'object', 'properties': {}}
    hyperparams_schema['relevantToOptimizer'] = []
    method_schemas = {'hyperparams': {'allOf': [hyperparams_schema]}}
    if hasattr(impl, 'fit'):
        method_schemas['input_fit'] = _get_args_schema(impl.fit)
    for method_name in ['predict', 'predict_proba', 'transform']:
        if hasattr(impl, method_name):
            method_args_schema = _get_args_schema(getattr(impl, method_name))
            method_schemas['input_' + method_name] = method_args_schema
            method_schemas['output_' + method_name] = {'laleType': 'Any'}
    result = {
        '$schema': 'http://json-schema.org/draft-04/schema#',
        'description':
        'Combined schema for expected data and hyperparameters.',
        'type': 'object',
        'properties': method_schemas}
    return result
