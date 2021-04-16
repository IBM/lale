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

from typing import Any, Optional, Tuple, Type

import numpy as np
import pandas as pd
import scipy.sparse

import lale.type_checking
from lale.type_checking import JSON_TYPE

try:
    import torch

    torch_installed = True
except ImportError:
    torch_installed = False


# See instructions for subclassing numpy ndarray:
# https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
class NDArrayWithSchema(np.ndarray):
    def __new__(
        cls,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        json_schema=None,
    ):
        result = super(NDArrayWithSchema, cls).__new__(
            cls, shape, dtype, buffer, offset, strides, order  # type: ignore
        )
        result.json_schema = json_schema
        return result

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.json_schema = getattr(obj, "json_schema", None)


# See instructions for subclassing pandas DataFrame:
# https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas
class DataFrameWithSchema(pd.DataFrame):
    _internal_names = pd.DataFrame._internal_names + ["json_schema"]
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        return DataFrameWithSchema


class SeriesWithSchema(pd.Series):
    _internal_names = pd.Series._internal_names + ["json_schema"]
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        return SeriesWithSchema


def add_schema(obj, schema=None, raise_on_failure=False, recalc=False):
    from lale.settings import disable_data_schema_validation

    if disable_data_schema_validation:
        return obj
    if obj is None:
        return None
    if isinstance(obj, NDArrayWithSchema):
        result = obj
    elif isinstance(obj, np.ndarray):
        result = obj.view(NDArrayWithSchema)
    elif isinstance(obj, SeriesWithSchema):
        result = obj
    elif isinstance(obj, pd.Series):
        result = SeriesWithSchema(obj)
    elif isinstance(obj, DataFrameWithSchema):
        result = obj
    elif isinstance(obj, pd.DataFrame):
        result = DataFrameWithSchema(obj)
    elif is_list_tensor(obj):
        obj = np.array(obj)
        result = obj.view(NDArrayWithSchema)
    elif raise_on_failure:
        raise ValueError(f"unexpected type(obj) {type(obj)}")
    else:
        return obj
    if recalc:
        result.json_schema = None
    if not hasattr(result, "json_schema") or result.json_schema is None:
        if schema is None:
            result.json_schema = to_schema(obj)
        else:
            lale.type_checking.validate_is_schema(schema)
            result.json_schema = schema
    return result


def add_schema_adjusting_n_rows(obj, schema):
    assert isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)), type(obj)
    assert schema.get("type", None) == "array", schema
    n_rows = obj.shape[0]
    mod_schema = {**schema, "minItems": n_rows, "maxItems": n_rows}
    result = add_schema(obj, mod_schema)
    return result


def strip_schema(obj):
    if isinstance(obj, NDArrayWithSchema):
        result = np.array(obj)
        assert type(result) == np.ndarray
    elif isinstance(obj, SeriesWithSchema):
        result = pd.Series(obj)
        assert type(result) == pd.Series
    elif isinstance(obj, DataFrameWithSchema):
        result = pd.DataFrame(obj)
        assert type(result) == pd.DataFrame
    else:
        result = obj
    return result


def dtype_to_schema(typ) -> JSON_TYPE:
    result: JSON_TYPE
    if typ is bool or np.issubdtype(typ, np.bool_):
        result = {"type": "boolean"}
    elif np.issubdtype(typ, np.unsignedinteger):
        result = {"type": "integer", "minimum": 0}
    elif np.issubdtype(typ, np.integer):
        result = {"type": "integer"}
    elif np.issubdtype(typ, np.number):
        result = {"type": "number"}
    elif np.issubdtype(typ, np.string_) or np.issubdtype(typ, np.unicode_):
        result = {"type": "string"}
    elif isinstance(typ, np.dtype):
        if typ.fields:
            props = {k: dtype_to_schema(t) for k, t in typ.fields.items()}
            result = {"type": "object", "properties": props}
        elif typ.shape:
            result = shape_and_dtype_to_schema(typ.shape, typ.subdtype)
        elif np.issubdtype(typ, np.object_):
            result = {"type": "string"}
        else:
            assert False, f"unexpected dtype {typ}"
    else:
        assert False, f"unexpected non-dtype {typ}"
    lale.type_checking.validate_is_schema(result)
    return result


def shape_and_dtype_to_schema(shape, dtype) -> JSON_TYPE:
    result = dtype_to_schema(dtype)
    for dim in reversed(shape):
        result = {"type": "array", "minItems": dim, "maxItems": dim, "items": result}
    lale.type_checking.validate_is_schema(result)
    return result


def list_tensor_to_shape_and_dtype(ls) -> Optional[Tuple[Tuple[int, ...], Type]]:
    if isinstance(ls, (int, float, str)):
        return ((), type(ls))
    if isinstance(ls, list):
        sub_result: Any = "Any"
        for item in ls:
            item_result = list_tensor_to_shape_and_dtype(item)
            if item_result is None:
                return None
            if sub_result == "Any":
                sub_result = item_result
            elif sub_result != item_result:
                return None
        if sub_result == "Any" and len(ls) == 0:
            return ((len(ls),) + (), int)
        sub_shape, sub_dtype = sub_result
        return ((len(ls),) + sub_shape, sub_dtype)
    return None


def is_list_tensor(obj) -> bool:
    if isinstance(obj, list):
        shape_and_dtype = list_tensor_to_shape_and_dtype(obj)
        return shape_and_dtype is not None
    return False


def list_tensor_to_schema(ls) -> Optional[JSON_TYPE]:
    shape_and_dtype = list_tensor_to_shape_and_dtype(ls)
    if shape_and_dtype is None:
        return None
    result = shape_and_dtype_to_schema(*shape_and_dtype)
    return result


def ndarray_to_schema(array) -> JSON_TYPE:
    assert isinstance(array, np.ndarray)
    if (
        isinstance(array, NDArrayWithSchema)
        and hasattr(array, "json_schema")
        and array.json_schema is not None
    ):
        return array.json_schema
    return shape_and_dtype_to_schema(array.shape, array.dtype)


def csr_matrix_to_schema(matrix) -> JSON_TYPE:
    assert isinstance(matrix, scipy.sparse.csr_matrix)
    result = shape_and_dtype_to_schema(matrix.shape, matrix.dtype)
    result["isSparse"] = {}  # true schema
    return result


def dataframe_to_schema(df) -> JSON_TYPE:
    assert isinstance(df, pd.DataFrame)
    if (
        isinstance(df, DataFrameWithSchema)
        and hasattr(df, "json_schema")
        and df.json_schema is not None
    ):
        return df.json_schema
    n_rows, n_columns = df.shape
    assert n_columns == len(df.columns) and n_columns == len(df.dtypes)
    items = [
        {"description": str(col), **dtype_to_schema(df.dtypes[col])}
        for col in df.columns
    ]
    result = {
        "type": "array",
        "minItems": n_rows,
        "maxItems": n_rows,
        "items": {
            "type": "array",
            "minItems": n_columns,
            "maxItems": n_columns,
            "items": items,
        },
    }
    lale.type_checking.validate_is_schema(result)
    return result


def series_to_schema(series) -> JSON_TYPE:
    assert isinstance(series, pd.Series)
    if (
        isinstance(series, SeriesWithSchema)
        and hasattr(series, "json_schema")
        and series.json_schema is not None
    ):
        return series.json_schema
    (n_rows,) = series.shape
    result = {
        "type": "array",
        "minItems": n_rows,
        "maxItems": n_rows,
        "items": {"description": str(series.name), **dtype_to_schema(series.dtype)},
    }
    lale.type_checking.validate_is_schema(result)
    return result


def torch_tensor_to_schema(tensor) -> JSON_TYPE:
    assert torch_installed, """Your Python environment does not have torch installed. You can install it with
    pip install torch
or with
    pip install 'lale[full]'"""
    assert isinstance(tensor, torch.Tensor)
    result: JSON_TYPE
    # https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
    if tensor.dtype == torch.bool:
        result = {"type": "boolean"}
    elif tensor.dtype == torch.uint8:
        result = {"type": "integer", "minimum": 0, "maximum": 255}
    elif torch.is_floating_point(tensor):
        result = {"type": "number"}
    else:
        result = {"type": "integer"}
    for dim in reversed(tensor.shape):
        result = {"type": "array", "minItems": dim, "maxItems": dim, "items": result}
    return result


def is_liac_arff(obj) -> bool:
    expected_types = {
        "description": str,
        "relation": str,
        "attributes": list,
        "data": list,
    }
    if not isinstance(obj, dict):
        return False
    for k, t in expected_types.items():
        if k not in obj or not isinstance(obj[k], t):
            return False
    return True


def liac_arff_to_schema(larff) -> JSON_TYPE:
    assert is_liac_arff(
        larff
    ), """Your Python environment might contain an 'arff' package different from 'liac-arff'. You can install it with
    pip install 'liac-arff>=2.4.0'
or with
    pip install 'lale[full]'"""
    n_rows, n_columns = len(larff["data"]), len(larff["attributes"])

    def larff_type_to_schema(larff_type) -> JSON_TYPE:
        if isinstance(larff_type, str):
            a2j = {
                "numeric": "number",
                "real": "number",
                "integer": "integer",
                "string": "string",
            }
            return {"type": a2j[larff_type.lower()]}
        assert isinstance(larff_type, list)
        return {"enum": [*larff_type]}

    items = [
        {"description": attr[0], **larff_type_to_schema(attr[1])}
        for attr in larff["attributes"]
    ]
    result = {
        "type": "array",
        "minItems": n_rows,
        "maxItems": n_rows,
        "items": {
            "type": "array",
            "minItems": n_columns,
            "maxItems": n_columns,
            "items": items,
        },
    }
    lale.type_checking.validate_is_schema(result)
    return result


def to_schema(obj) -> JSON_TYPE:
    result = None
    if obj is None:
        result = {"enum": [None]}
    elif isinstance(obj, np.ndarray):
        result = ndarray_to_schema(obj)
    elif isinstance(obj, scipy.sparse.csr_matrix):
        result = csr_matrix_to_schema(obj)
    elif isinstance(obj, pd.DataFrame):
        result = dataframe_to_schema(obj)
    elif isinstance(obj, pd.Series):
        result = series_to_schema(obj)
    elif torch_installed and isinstance(obj, torch.Tensor):
        result = torch_tensor_to_schema(obj)
    elif is_liac_arff(obj):
        result = liac_arff_to_schema(obj)
    elif lale.type_checking.is_schema(obj):
        result = obj
    elif isinstance(obj, list):
        result = list_tensor_to_schema(obj)
    if result is None:
        raise ValueError(f"to_schema(obj), type {type(obj)}, value {obj}")
    lale.type_checking.validate_is_schema(result)
    return result
