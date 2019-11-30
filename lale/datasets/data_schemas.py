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

import lale.helpers
import numpy as np
import pandas as pd

# See instructions for subclassing numpy ndarray:
# https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
class NDArrayWithSchema(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, json_schema=None):
        result = super(NDArrayWithSchema, subtype).__new__(
            subtype, shape, dtype, buffer, offset, strides, order)
        result.json_schema = json_schema
        return result

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.json_schema = getattr(obj, 'json_schema', None)

# See instructions for subclassing pandas DataFrame:
# https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas
class DataFrameWithSchema(pd.DataFrame):
    _internal_names = pd.DataFrame._internal_names + ['json_schema']
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        return DataFrameWithSchema

class SeriesWithSchema(pd.Series):
    _internal_names = pd.Series._internal_names + ['json_schema']
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        return SeriesWithSchema

def add_schema(obj, schema):
    lale.helpers.validate_is_schema(schema)
    if isinstance(obj, np.ndarray):
        result = obj.view(NDArrayWithSchema)
        result.json_schema = schema
    elif isinstance(obj, pd.Series):
        result = SeriesWithSchema(obj)
        result.json_schema = schema
    elif isinstance(obj, pd.DataFrame):
        result = DataFrameWithSchema(obj)
        result.json_schema = schema
    else:
        assert False, f'unexpected type(obj) {type(obj)}'
    assert result.json_schema == schema
    return result

def dtype_to_schema(typ):
    result = None
    if typ is bool or typ is np.bool_:
        result = {'type': 'boolean'}
    elif np.issubdtype(typ, np.unsignedinteger):
        result = {'type': 'integer', 'minimum': 0}
    elif np.issubdtype(typ, np.integer):
        result = {'type': 'integer'}
    elif np.issubdtype(typ, np.number):
        result = {'type': 'number'}
    elif np.issubdtype(typ, np.string_):
        result = {'type': 'string'}
    elif isinstance(typ, np.dtype):
        if typ.fields:
            props = {k: dtype_to_schema(t) for k, t in typ.fields.items()}
            result = {'type': 'object', 'properties': props}
        elif typ.shape:
            result = shape_and_dtype_to_schema(typ.shape, typ.subdtype)
        elif np.issubdtype(typ, np.object_):
            result = {'type': 'string'}
        else:
            assert False, f'unexpected dtype {typ}'
    else:
        assert False, f'unexpected non-dtype {typ}'
    lale.helpers.validate_is_schema(result)
    return result

def shape_and_dtype_to_schema(shape, dtype):
    result = dtype_to_schema(dtype)
    for dim in reversed(shape):
        result = {
            'type': 'array',
            'minItems': dim,
            'maxItems': dim,
            'items': result}
    lale.helpers.validate_is_schema(result)
    return result

def ndarray_to_schema(array):
    assert isinstance(array, np.ndarray)
    if isinstance(array, NDArrayWithSchema) and hasattr(array, 'json_schema'):
        return array.json_schema
    return shape_and_dtype_to_schema(array.shape, array.dtype)

def dataframe_to_schema(df):
    assert isinstance(df, pd.DataFrame)
    if isinstance(df, DataFrameWithSchema) and hasattr(df, 'json_schema'):
        return df.json_schema
    n_rows, n_columns = df.shape
    assert n_columns == len(df.columns) and n_columns == len(df.dtypes)
    items = [
        {'description': str(col), **dtype_to_schema(df.dtypes[col])}
        for col in df.columns]
    result = {
        'type': 'array',
        'minItems': n_rows,
        'maxItems': n_rows,
        'items': {
            'type': 'array',
            'minItems': n_columns,
            'maxItems': n_columns,
            'items': items}}
    lale.helpers.validate_is_schema(result)
    return result

def series_to_schema(series):
    assert isinstance(series, pd.Series)
    if isinstance(series, SeriesWithSchema) and hasattr(series, 'json_schema'):
        return series.json_schema
    (n_rows, ) = series.shape
    result = {
        'type': 'array',
        'minItems': n_rows,
        'maxItems': n_rows,
        'items': {
            'description': str(series.name),
            **dtype_to_schema(series.dtype)}}
    lale.helpers.validate_is_schema(result)
    return result

def is_liac_arff(obj):
    expected_types = {
        'description': str, 'relation': str, 'attributes': list, 'data': list}
    if not isinstance(obj, dict):
        return False
    for k, t in expected_types.items():
        if k not in obj or not isinstance(obj[k], t):
            return False
    return True

def liac_arff_to_schema(larff):
    assert is_liac_arff(larff), 'Your Python environment might contain the wrong arff library, this code requires liac-arff.'
    n_rows, n_columns = len(larff['data']), len(larff['attributes'])
    def larff_type_to_schema(larff_type):
        if isinstance(larff_type, str):
            a2j = {'numeric': 'number', 'real': 'number',
                   'integer': 'integer', 'string': 'string'}
            return {'type': a2j[larff_type.lower()]}
        assert isinstance(larff_type, list)
        return {'enum': [*larff_type]}
    items = [
        {'description': attr[0], **larff_type_to_schema(attr[1])}
        for attr in larff['attributes']]
    result = {
        'type': 'array',
        'minItems': n_rows,
        'maxItems': n_rows,
        'items': {
            'type': 'array',
            'minItems': n_columns,
            'maxItems': n_columns,
            'items': items}}
    lale.helpers.validate_is_schema(result)
    return result

def to_schema(obj):
    if isinstance(obj, np.ndarray):
        result = ndarray_to_schema(obj)
    elif isinstance(obj, pd.DataFrame):
        result = dataframe_to_schema(obj)
    elif isinstance(obj, pd.Series):
        result = series_to_schema(obj)
    elif is_liac_arff(obj):
        result = liac_arff_to_schema(obj)
    else:
        result = dtype_to_schema(obj)
    result = {'$schema': lale.helpers.JSON_META_SCHEMA_URL, **result}
    lale.helpers.validate_is_schema(result)
    return result
