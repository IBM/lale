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
import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, overload

import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import jsonsubschema

import lale.helpers

if TYPE_CHECKING:
    import lale.operators

JSON_TYPE = Dict[str, Any]


def _validate_lale_type(validator, laleType, instance, schema):
    # https://github.com/Julian/jsonschema/blob/master/jsonschema/_validators.py
    if laleType == "Any":
        return
    elif laleType == "callable":
        if not callable(instance):
            yield jsonschema.exceptions.ValidationError(
                f"expected {laleType}, got {type(instance)}"
            )
    elif laleType == "operator":
        import sklearn.base

        import lale.operators

        if not (
            isinstance(instance, lale.operators.Operator)
            or isinstance(instance, sklearn.base.BaseEstimator)
            or (
                inspect.isclass(instance)
                and issubclass(instance, sklearn.base.BaseEstimator)
            )
        ):
            yield jsonschema.exceptions.ValidationError(
                f"expected {laleType}, got {type(instance)}"
            )
    elif laleType == "expression":
        import lale.expressions

        if not isinstance(instance, lale.expressions.Expr):
            yield jsonschema.exceptions.ValidationError(
                f"expected {laleType}, got {type(instance)}"
            )
    elif laleType == "numpy.random.RandomState":
        import numpy.random

        if not isinstance(instance, numpy.random.RandomState):
            yield jsonschema.exceptions.ValidationError(
                f"expected {laleType}, got {type(instance)}"
            )


# https://github.com/Julian/jsonschema/blob/master/jsonschema/validators.py
_lale_validator = jsonschema.validators.extend(
    validator=jsonschema.Draft4Validator, validators={"laleType": _validate_lale_type}
)


def always_validate_schema(value, schema: JSON_TYPE, subsample_array: bool = True):
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
    jsonschema.validate(
        json_value, lale.helpers.data_to_json(schema, False), _lale_validator
    )


def validate_schema(value, schema: JSON_TYPE, subsample_array: bool = True):
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
    from lale.settings import disable_hyperparams_schema_validation

    if disable_hyperparams_schema_validation:
        return True  # if schema validation is disabled, always return as valid
    return always_validate_schema(value, schema, subsample_array=subsample_array)


_JSON_META_SCHEMA_URL = "http://json-schema.org/draft-04/schema#"


def _json_meta_schema() -> Dict[str, Any]:
    return jsonschema.Draft4Validator.META_SCHEMA


def validate_is_schema(value: Dict[str, Any]):
    # only checking hyperparams schema validation flag because it is likely to be true and this call is cheap.
    from lale.settings import disable_hyperparams_schema_validation

    if disable_hyperparams_schema_validation:
        return True

    if "$schema" in value:
        assert value["$schema"] == _JSON_META_SCHEMA_URL
    jsonschema.validate(value, _json_meta_schema())


def is_schema(value) -> bool:
    if isinstance(value, dict):
        try:
            jsonschema.validate(value, _json_meta_schema())
        except jsonschema.ValidationError:
            return False
        return True
    return False


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
                return new
        result = {k: _json_replace(v, old, new) for k, v in subject.items()}
        for k in subject:
            if subject[k] != result[k]:
                return result
    return subject  # nothing changed so share original object (not a copy)


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
    new_sub = _json_replace(sub_schema, {"laleType": "Any"}, {"not": {}})
    try:
        return jsonsubschema.isSubschema(new_sub, super_schema)
    except Exception as e:
        raise ValueError(
            f"unexpected internal error checking ({new_sub} <: {super_schema})"
        ) from e


class SubschemaError(Exception):
    """Raised when a subschema check (sub `<:` sup) failed."""

    def __init__(self, sub, sup, sub_name="sub", sup_name="super"):
        self.sub = sub
        self.sup = sup
        self.sub_name = sub_name
        self.sup_name = sup_name

    def __str__(self):
        summary = f"Expected {self.sub_name} to be a subschema of {self.sup_name}."
        import lale.pretty_print

        sub = lale.pretty_print.json_to_string(self.sub)
        sup = lale.pretty_print.json_to_string(self.sup)
        details = f"\n{self.sub_name} = {sub}\n{self.sup_name} = {sup}"
        return summary + details


def _validate_subschema(
    sub: JSON_TYPE, sup: JSON_TYPE, sub_name="sub", sup_name="super"
):
    if not is_subschema(sub, sup):
        raise SubschemaError(sub, sup, sub_name, sup_name)


def validate_schema_or_subschema(lhs: Any, super_schema: JSON_TYPE):
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
    from lale.settings import disable_data_schema_validation

    if disable_data_schema_validation:
        return True  # If schema validation is disabled, always return as valid
    sub_schema: Optional[JSON_TYPE]
    if is_schema(lhs):
        sub_schema = lhs
    else:
        import lale.datasets.data_schemas

        try:
            sub_schema = lale.datasets.data_schemas.to_schema(lhs)
        except ValueError:
            sub_schema = None
    if sub_schema is None:
        validate_schema(lhs, super_schema)
    else:
        _validate_subschema(sub_schema, super_schema)


def join_schemas(*schemas: JSON_TYPE) -> JSON_TYPE:
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

    def join_two_schemas(s_a: JSON_TYPE, s_b: JSON_TYPE) -> JSON_TYPE:
        if s_a is None:
            return s_b
        s_a = lale.helpers.dict_without(s_a, "description")
        s_b = lale.helpers.dict_without(s_b, "description")
        if is_subschema(s_a, s_b):
            return s_b
        if is_subschema(s_b, s_a):
            return s_a
        # we should improve the typing of the jsonsubschema API so that this ignore can be removed
        return jsonsubschema.joinSchemas(s_a, s_b)  # type: ignore

    if len(schemas) == 0:
        return {"not": {}}
    result = functools.reduce(join_two_schemas, schemas)
    return result


def get_hyperparam_names(op: "lale.operators.IndividualOp") -> List[str]:
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
    if op.impl_class.__module__.startswith("lale"):
        hp_schema = op.hyperparam_schema()
        params = next(iter(hp_schema.get("allOf", []))).get("properties", {})
        return list(params.keys())
    else:
        c: Any = op.impl_class
        return inspect.getargspec(c.__init__).args


def validate_method(op: "lale.operators.IndividualOp", schema_name: str):
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
    if op._impl.__module__.startswith("lale"):
        assert schema_name in op._schemas["properties"]
    else:
        method_name = ""
        if schema_name.startswith("input_"):
            method_name = schema_name[len("input_") :]
        elif schema_name.startswith("output_"):
            method_name = schema_name[len("output_") :]
        if method_name:
            assert hasattr(op._impl, method_name)


def _get_args_schema(fun):
    sig = inspect.signature(fun)
    result = {"type": "object", "properties": {}}
    required = []
    additional_properties = False
    for name, param in sig.parameters.items():
        ignored_kinds = [
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ]
        if name != "self":
            if param.kind in ignored_kinds:
                additional_properties = True
            else:
                if param.default == inspect.Parameter.empty:
                    param_schema = {"laleType": "Any"}
                    required.append(name)
                else:
                    param_schema = {"default": param.default}
                result["properties"][name] = param_schema
    if not additional_properties:
        result["additionalProperties"] = False
    if len(required) > 0:
        result["required"] = required
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
    if hasattr(impl, "__init__"):
        hyperparams_schema = _get_args_schema(impl.__init__)
    else:
        hyperparams_schema = {"type": "object", "properties": {}}
    hyperparams_schema["relevantToOptimizer"] = []
    method_schemas: Dict[str, JSON_TYPE] = {
        "hyperparams": {"allOf": [hyperparams_schema]}
    }
    if hasattr(impl, "fit"):
        method_schemas["input_fit"] = _get_args_schema(impl.fit)
    for method_name in ["predict", "predict_proba", "transform"]:
        if hasattr(impl, method_name):
            method_args_schema = _get_args_schema(getattr(impl, method_name))
            method_schemas["input_" + method_name] = method_args_schema
            method_schemas["output_" + method_name] = {"laleType": "Any"}
    tags = {
        "pre": [],
        "op": (["transformer"] if hasattr(impl, "transform") else [])
        + (["estimator"] if hasattr(impl, "predict") else []),
        "post": [],
    }
    result = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "description": f"Schema for {type(impl)} auto-generated by lale.type_checking.get_default_schema().",
        "type": "object",
        "tags": tags,
        "properties": method_schemas,
    }
    return result


_data_info_keys = {"laleMaximum": "maximum", "laleNot": "not"}


def has_data_constraints(hyperparam_schema: JSON_TYPE) -> bool:
    def recursive_check(subject: Any) -> bool:
        if isinstance(subject, (list, tuple)):
            for v in subject:
                if recursive_check(v):
                    return True
        elif isinstance(subject, dict):
            for k, v in subject.items():
                if k in _data_info_keys or recursive_check(v):
                    return True
        return False

    result = recursive_check(hyperparam_schema)
    return result


def replace_data_constraints(
    hyperparam_schema: JSON_TYPE, data_schema: JSON_TYPE
) -> JSON_TYPE:
    @overload
    def recursive_replace(subject: JSON_TYPE) -> JSON_TYPE:
        ...

    @overload
    def recursive_replace(subject: List) -> List:
        ...

    @overload
    def recursive_replace(subject: Tuple) -> Tuple:
        ...

    @overload
    def recursive_replace(subject: Any) -> Any:
        ...

    def recursive_replace(subject):
        any_changes = False
        if isinstance(subject, (list, tuple)):
            result = []
            for v in subject:
                new_v = recursive_replace(v)
                result.append(new_v)
                any_changes = any_changes or v is not new_v
            if isinstance(subject, tuple):
                result = tuple(result)
        elif isinstance(subject, dict):
            result = {}
            for k, v in subject.items():
                if k in _data_info_keys:
                    new_v = lale.helpers.json_lookup("properties/" + v, data_schema)
                    if new_v is None:
                        new_k = k
                        new_v = v
                    else:
                        new_k = _data_info_keys[k]
                else:
                    new_v = recursive_replace(v)
                    new_k = k
                result[new_k] = new_v
                any_changes = any_changes or k != new_k or v is not new_v
        else:
            return subject
        return result if any_changes else subject

    result = recursive_replace(hyperparam_schema)
    return result
