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

import os

import numpy as np
import pandas as pd
import scipy.sparse

import lale.type_checking

try:
    import torch

    torch_installed = True
except ImportError:
    torch_installed = False


# See instructions for subclassing numpy ndarray:
# https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
class NDArrayWithSchema(np.ndarray):
    def __new__(
        subtype,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        json_schema=None,
    ):
        result = super(NDArrayWithSchema, subtype).__new__(
            subtype, shape, dtype, buffer, offset, strides, order
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


def is_list_tensor(obj):
    def list_tensor_shape(ls):
        if isinstance(ls, int) or isinstance(ls, float) or isinstance(ls, str):
            return (str(type(ls)),)
        if isinstance(ls, list):
            sub_shape = "Any"
            for item in ls:
                item_result = list_tensor_shape(item)
                if item_result is None:
                    return None
                if sub_shape == "Any":
                    sub_shape = item_result
                elif sub_shape != item_result:
                    return None
            return (len(ls),) + sub_shape
        return None

    if isinstance(obj, list):
        shape = list_tensor_shape(obj)
        return shape is not None
    return False


def add_schema(obj, schema=None, raise_on_failure=False, recalc=False):
    disable_schema = os.environ.get("LALE_DISABLE_SCHEMA_VALIDATION", None)
    if disable_schema is not None and disable_schema.lower() == "true":
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


def dtype_to_schema(typ):
    result = None
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


def shape_and_dtype_to_schema(shape, dtype):
    result = dtype_to_schema(dtype)
    for dim in reversed(shape):
        result = {"type": "array", "minItems": dim, "maxItems": dim, "items": result}
    lale.type_checking.validate_is_schema(result)
    return result


def ndarray_to_schema(array):
    assert isinstance(array, np.ndarray)
    if (
        isinstance(array, NDArrayWithSchema)
        and hasattr(array, "json_schema")
        and array.json_schema is not None
    ):
        return array.json_schema
    return shape_and_dtype_to_schema(array.shape, array.dtype)


def csr_matrix_to_schema(matrix):
    assert isinstance(matrix, scipy.sparse.csr_matrix)
    result = shape_and_dtype_to_schema(matrix.shape, matrix.dtype)
    result["isSparse"] = {}  # true schema
    return result


def dataframe_to_schema(df):
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


def series_to_schema(series):
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


def torch_tensor_to_schema(tensor):
    assert torch_installed, """Your Python environment does not have torch installed. You can install it with
    pip install torch
or with
    pip install 'lale[full]'"""
    assert isinstance(tensor, torch.Tensor)
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


def is_liac_arff(obj):
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


def liac_arff_to_schema(larff):
    assert is_liac_arff(
        larff
    ), """Your Python environment might contain an 'arff' package different from 'liac-arff'. You can install it with
    pip install 'liac-arff>=2.4.0'
or with
    pip install 'lale[full]'"""
    n_rows, n_columns = len(larff["data"]), len(larff["attributes"])

    def larff_type_to_schema(larff_type):
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


def to_schema(obj):
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
    else:
        raise ValueError(f"to_schema(obj), type {type(obj)}, value {obj}")
    lale.type_checking.validate_is_schema(result)
    return result
