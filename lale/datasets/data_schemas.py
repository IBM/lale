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

from typing import Any, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from numpy import issubdtype, ndarray
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from scipy.sparse import csr_matrix

import lale.type_checking
from lale.helpers import _is_spark_df
from lale.type_checking import JSON_TYPE

try:
    import torch
    from torch import Tensor
except ImportError:
    torch = None
    Tensor = None

try:
    from py4j.protocol import Py4JError
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import GroupedData as SparkGroupedData

except ImportError:
    Py4JError = None
    SparkDataFrame = None
    SparkGroupedData = None


# See instructions for subclassing numpy ndarray:
# https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
class NDArrayWithSchema(ndarray):
    def __new__(
        cls,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        json_schema=None,
        table_name=None,
    ):
        result = super(  # pylint:disable=too-many-function-args
            NDArrayWithSchema, cls
        ).__new__(
            cls, shape, dtype, buffer, offset, strides, order  # type: ignore
        )
        result.json_schema = json_schema
        result.table_name = table_name
        return result

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.json_schema = getattr(obj, "json_schema", None)
        self.table_name = getattr(obj, "table_name", None)


# See instructions for subclassing pandas DataFrame:
# https://pandas.pydata.org/pandas-docs/stable/development/extending.html#extending-subclassing-pandas
class DataFrameWithSchema(DataFrame):
    _internal_names = DataFrame._internal_names + ["json_schema", "table_name"]
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        return DataFrameWithSchema


class SeriesWithSchema(Series):
    _internal_names = DataFrame._internal_names + [
        "json_schema",
        "table_name",
        "folds_for_monoid",
    ]
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        return SeriesWithSchema


if SparkDataFrame is not None:

    def _gen_index_name(df, cpt=None):
        name = f"index{cpt if cpt is not None else ''}"
        if name in df.columns:
            return _gen_index_name(df, cpt=cpt + 1 if cpt is not None else 0)
        else:
            return name

    class SparkDataFrameWithIndex(SparkDataFrame):  # type: ignore
        def __init__(self, df, index_names=None):
            if index_names is not None and len(index_names) == 1:
                index_name = index_names[0]
            elif index_names is None or len(index_names) == 0:
                index_name = _gen_index_name(df)
                index_names = [index_name]
            else:
                index_name = None
            if index_name is not None and index_name not in df.columns:
                df_with_index = (
                    df.rdd.zipWithIndex()
                    .map(lambda row: row[0] + (row[1],))
                    .toDF(df.columns + [index_name])
                )
            else:
                df_with_index = df
            table_name = get_table_name(df)
            if table_name is not None:
                df_with_index = df_with_index.alias(table_name)
            super().__init__(df_with_index._jdf, df_with_index.sql_ctx)
            self.index_name = index_name
            self.index_names = index_names
            for f in df.schema.fieldNames():
                self.schema[f].metadata = df.schema[f].metadata

        def drop_indexes(self):
            result = self.drop(*self.index_names)
            result = add_table_name(result, get_table_name(self))
            return result

        @property
        def columns_without_indexes(self):
            cols = list(self.columns)
            for name in self.index_names:
                cols.remove(name)
            return cols

        def toPandas(self, *args, **kwargs):
            df = super().toPandas(*args, **kwargs)
            return df.set_index(self.index_names)

else:

    class SparkDataFrameWithIndex:  # type: ignore
        def __init__(self, df, index_names=None) -> None:
            raise ValueError("pyspark is not installed")  # type: ignore

        @property
        def index_name(self) -> Union[str, None]:
            raise ValueError("pyspark is not installed")  # type: ignore

        @property
        def index_names(self) -> List[str]:
            raise ValueError("pyspark is not installed")  # type: ignore

        def toPandas(self, *args, **kwargs) -> DataFrame:
            raise ValueError("pyspark is not installed")  # type: ignore

        @property
        def schema(self) -> Any:
            raise ValueError("pyspark is not installed")  # type: ignore


def add_schema(obj, schema=None, raise_on_failure=False, recalc=False) -> Any:
    from lale.settings import disable_data_schema_validation

    if disable_data_schema_validation:
        return obj
    if obj is None:
        return None
    if isinstance(obj, NDArrayWithSchema):
        result = obj
    elif isinstance(obj, ndarray):
        result = obj.view(NDArrayWithSchema)
    elif isinstance(obj, SeriesWithSchema):
        result = obj
    elif isinstance(obj, Series):
        result = SeriesWithSchema(obj)
    elif isinstance(obj, DataFrameWithSchema):
        result = obj
    elif isinstance(obj, DataFrame):
        result = DataFrameWithSchema(obj)
    elif is_list_tensor(obj):
        obj = np.array(obj)
        result = obj.view(NDArrayWithSchema)
    elif raise_on_failure:
        raise ValueError(f"unexpected type(obj) {type(obj)}")
    else:
        return obj
    if recalc:
        setattr(result, "json_schema", None)
    if getattr(result, "json_schema", None) is None:
        if schema is None:
            setattr(result, "json_schema", to_schema(obj))
        else:
            lale.type_checking.validate_is_schema(schema)
            setattr(result, "json_schema", schema)
    return result


def add_schema_adjusting_n_rows(obj, schema):
    assert isinstance(obj, (ndarray, DataFrame, Series)), type(obj)
    assert schema.get("type", None) == "array", schema
    n_rows = obj.shape[0]
    mod_schema = {**schema, "minItems": n_rows, "maxItems": n_rows}
    result = add_schema(obj, mod_schema)
    return result


def add_table_name(obj, name) -> Any:
    if obj is None:
        return None
    if name is None:
        return obj
    if SparkDataFrame is not None and isinstance(obj, SparkDataFrame):
        # alias method documentation: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.alias.html
        # Python class DataFrame with method alias(self, alias): https://github.com/apache/spark/blob/master/python/pyspark/sql/dataframe.py
        # Scala type DataFrame: https://github.com/apache/spark/blob/master/sql/core/src/main/scala/org/apache/spark/sql/package.scala
        # Scala class DataSet with method as(alias: String): https://github.com/apache/spark/blob/master/sql/core/src/main/scala/org/apache/spark/sql/Dataset.scala
        o = obj.alias(name)
        for f in obj.schema.fieldNames():
            o.schema[f].metadata = obj.schema[f].metadata
        if isinstance(obj, SparkDataFrameWithIndex):
            o = SparkDataFrameWithIndex(o, obj.index_names)
        return o
    if isinstance(obj, NDArrayWithSchema):
        result = obj.view(NDArrayWithSchema)
        if hasattr(obj, "json_schema"):
            result.json_schema = obj.json_schema
    elif isinstance(obj, ndarray):
        result = obj.view(NDArrayWithSchema)
    elif isinstance(obj, SeriesWithSchema):
        result = obj.copy(deep=False)
        if hasattr(obj, "json_schema"):
            result.json_schema = obj.json_schema
    elif isinstance(obj, Series):
        result = SeriesWithSchema(obj)
    elif isinstance(obj, DataFrameWithSchema):
        result = obj.copy(deep=False)
        if hasattr(obj, "json_schema"):
            result.json_schema = obj.json_schema
    elif isinstance(obj, DataFrame):
        result = DataFrameWithSchema(obj)
    elif is_list_tensor(obj):
        obj = np.array(obj)
        result = obj.view(NDArrayWithSchema)
    elif isinstance(obj, (DataFrameGroupBy, SeriesGroupBy)):
        result = obj
    elif SparkGroupedData is not None and isinstance(obj, SparkGroupedData):
        result = obj
    else:
        raise ValueError(f"unexpected type(obj) {type(obj)}")
    setattr(result, "table_name", name)
    return result


def get_table_name(obj):
    if (
        SparkDataFrame is not None
        and Py4JError is not None
        and isinstance(obj, SparkDataFrame)
    ):
        # Python class DataFrame with field self._jdf: https://github.com/apache/spark/blob/master/python/pyspark/sql/dataframe.py
        # Scala type DataFrame: https://github.com/apache/spark/blob/master/sql/core/src/main/scala/org/apache/spark/sql/package.scala
        # Scala class DataSet with field queryExecution: https://github.com/apache/spark/blob/master/sql/core/src/main/scala/org/apache/spark/sql/Dataset.scala
        # Scala fields turn into Java nullary methods
        # Py4J exposes Java methods as Python methods
        # Scala class QueryExecution with field analyzed: LogicalPlan: https://github.com/apache/spark/blob/master/sql/core/src/main/scala/org/apache/spark/sql/execution/QueryExecution.scala
        spark_query = obj._jdf.queryExecution().analyzed()  # type: ignore
        try:
            # calling spark_df.explain("extended") shows the analyzed contents
            # after spark_df.alias("foo"), analyzed contents should be SubqueryAlias
            # Scala class SuqueryAlias with field identifier: https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/plans/logical/basicLogicalOperators.scala
            # str(..) converts the Java string into a Python string
            result = str(spark_query.identifier())
        except Py4JError:
            result = None
        return result
    if isinstance(
        obj,
        (
            NDArrayWithSchema,
            SeriesWithSchema,
            DataFrameWithSchema,
            DataFrameGroupBy,
            SeriesGroupBy,
        ),
    ) or (SparkGroupedData is not None and isinstance(obj, SparkGroupedData)):
        return getattr(obj, "table_name", None)
    return None


def get_index_name(obj):
    result = None
    if SparkDataFrame is not None and isinstance(obj, SparkDataFrameWithIndex):
        result = obj.index_name
    elif isinstance(
        obj,
        (
            SeriesWithSchema,
            DataFrameWithSchema,
            DataFrameGroupBy,
            SeriesGroupBy,
        ),
    ):
        result = obj.index.name
    return result


def get_index_names(obj) -> Optional[list[str]]:
    result: Optional[list[str]] = None
    if SparkDataFrame is not None and isinstance(obj, SparkDataFrameWithIndex):
        result = obj.index_names
    elif isinstance(
        obj,
        (
            SeriesWithSchema,
            DataFrameWithSchema,
            DataFrameGroupBy,
            SeriesGroupBy,
        ),
    ):
        result = obj.index.names
    return result


def forward_metadata(old, new):
    new = add_table_name(new, get_table_name(old))
    if isinstance(old, SparkDataFrameWithIndex):
        new = SparkDataFrameWithIndex(new, index_names=get_index_names(old))
    return new


def strip_schema(obj):
    if isinstance(obj, NDArrayWithSchema):
        result = np.array(obj)
        assert type(result) is ndarray  # pylint:disable=unidiomatic-typecheck
    elif isinstance(obj, SeriesWithSchema):
        result = Series(obj)
        assert type(result) is Series  # pylint:disable=unidiomatic-typecheck
    elif isinstance(obj, DataFrameWithSchema):
        result = DataFrame(obj)
        assert type(result) is DataFrame  # pylint:disable=unidiomatic-typecheck
    else:
        result = obj
    return result


def _dtype_to_schema(typ) -> JSON_TYPE:
    result: JSON_TYPE
    if typ is bool or issubdtype(typ, np.bool_):
        result = {"type": "boolean"}
    elif issubdtype(typ, np.unsignedinteger):
        result = {"type": "integer", "minimum": 0}
    elif issubdtype(typ, np.integer):
        result = {"type": "integer"}
    elif issubdtype(typ, np.number):
        result = {"type": "number"}
    elif issubdtype(typ, np.str_) or issubdtype(typ, np.bytes_):
        result = {"type": "string"}
    elif isinstance(typ, np.dtype):
        if typ.fields:
            props = {k: _dtype_to_schema(t) for k, t in typ.fields.items()}
            result = {"type": "object", "properties": props}
        elif typ.shape:
            result = _shape_and_dtype_to_schema(typ.shape, typ.subdtype)
        elif issubdtype(typ, np.object_):
            result = {"type": "string"}
        else:
            assert False, f"unexpected dtype {typ}"
    else:
        assert False, f"unexpected non-dtype {typ}"
    return result


def dtype_to_schema(typ) -> JSON_TYPE:
    result = _dtype_to_schema(typ)
    lale.type_checking.validate_is_schema(result)
    return result


def _shape_and_dtype_to_schema(shape, dtype) -> JSON_TYPE:
    result = _dtype_to_schema(dtype)
    for dim in reversed(shape):
        result = {"type": "array", "minItems": dim, "maxItems": dim, "items": result}
    return result


def shape_and_dtype_to_schema(shape, dtype) -> JSON_TYPE:
    result = _shape_and_dtype_to_schema(shape, dtype)
    lale.type_checking.validate_is_schema(result)
    return result


def list_tensor_to_shape_and_dtype(ls) -> Optional[Tuple[Tuple[int, ...], Type]]:
    if isinstance(ls, (int, float, str)):
        return ((), type(ls))
    if isinstance(ls, list):
        sub_result: Union[Tuple[Tuple[int, ...], Type], Literal["Any"]] = "Any"
        for item in ls:
            item_result = list_tensor_to_shape_and_dtype(item)
            if item_result is None:
                return None
            if sub_result == "Any":
                sub_result = item_result
            elif sub_result != item_result:
                return None
        if sub_result == "Any":
            if len(ls) == 0:
                return ((len(ls),) + (), int)
            else:
                return None
        sub_shape, sub_dtype = sub_result
        return ((len(ls),) + sub_shape, sub_dtype)
    return None


def is_list_tensor(obj) -> bool:
    if isinstance(obj, list):
        shape_and_dtype = list_tensor_to_shape_and_dtype(obj)
        return shape_and_dtype is not None
    return False


def _list_tensor_to_schema(ls) -> Optional[JSON_TYPE]:
    shape_and_dtype = list_tensor_to_shape_and_dtype(ls)
    if shape_and_dtype is None:
        return None
    result = _shape_and_dtype_to_schema(*shape_and_dtype)
    return result


def list_tensor_to_schema(ls) -> Optional[JSON_TYPE]:
    result = _list_tensor_to_schema(ls)
    if result is None:
        return None
    lale.type_checking.validate_is_schema(result)
    return result


def _ndarray_to_schema(array) -> JSON_TYPE:
    assert isinstance(array, ndarray)
    if (
        isinstance(array, NDArrayWithSchema)
        and hasattr(array, "json_schema")
        and array.json_schema is not None
    ):
        return array.json_schema
    return _shape_and_dtype_to_schema(array.shape, array.dtype)


def ndarray_to_schema(array) -> JSON_TYPE:
    result = _ndarray_to_schema(array)
    lale.type_checking.validate_is_schema(result)
    return result


def _csr_matrix_to_schema(matrix) -> JSON_TYPE:
    assert isinstance(matrix, csr_matrix)
    result = _shape_and_dtype_to_schema(matrix.shape, matrix.dtype)
    result["isSparse"] = {}  # true schema
    return result


def csr_matrix_to_schema(matrix) -> JSON_TYPE:
    result = _csr_matrix_to_schema(matrix)
    lale.type_checking.validate_is_schema(result)
    return result


def _dataframe_to_schema(df) -> JSON_TYPE:
    assert isinstance(df, DataFrame)
    if (
        isinstance(df, DataFrameWithSchema)
        and hasattr(df, "json_schema")
        and df.json_schema is not None
    ):
        return df.json_schema
    n_rows, n_columns = df.shape
    df_dtypes = df.dtypes
    assert n_columns == len(df.columns) and n_columns == len(df_dtypes)
    items = [
        {"description": str(col), **_dtype_to_schema(df_dtypes[col])}
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
    return result


def dataframe_to_schema(df) -> JSON_TYPE:
    result = _dataframe_to_schema(df)
    lale.type_checking.validate_is_schema(result)
    return result


def _series_to_schema(series) -> JSON_TYPE:
    assert isinstance(series, Series)
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
        "items": {"description": str(series.name), **_dtype_to_schema(series.dtype)},
    }
    return result


def series_to_schema(series) -> JSON_TYPE:
    result = _series_to_schema(series)
    lale.type_checking.validate_is_schema(result)
    return result


def _torch_tensor_to_schema(tensor) -> JSON_TYPE:
    assert (
        torch is not None and Tensor is not None
    ), """Your Python environment does not have torch installed. You can install it with
    pip install torch
or with
    pip install 'lale[full]'"""
    assert isinstance(tensor, Tensor)
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


def torch_tensor_to_schema(tensor) -> JSON_TYPE:
    result = _torch_tensor_to_schema(tensor)
    lale.type_checking.validate_is_schema(result)
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


def _liac_arff_to_schema(larff) -> JSON_TYPE:
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
    return result


def liac_arff_to_schema(larff) -> JSON_TYPE:
    result = _liac_arff_to_schema(larff)
    lale.type_checking.validate_is_schema(result)
    return result


def make_optional_schema(schema: JSON_TYPE) -> JSON_TYPE:
    return {"anyOf": [schema, {"enum": [None]}]}


def _spark_df_to_schema(df) -> JSON_TYPE:
    assert (
        SparkDataFrame is not None
    ), """Your Python environment does not have spark installed. You can install it with
        pip install pyspark
"""
    assert isinstance(df, SparkDataFrameWithIndex)

    import pyspark.sql.types as stypes
    from pyspark.sql.types import StructField, StructType

    def maybe_make_optional(schema: JSON_TYPE, is_option: bool) -> JSON_TYPE:
        if is_option:
            return make_optional_schema(schema)
        return schema

    def spark_datatype_to_json_schema(dtype: stypes.DataType) -> JSON_TYPE:
        if isinstance(dtype, stypes.ArrayType):
            return {
                "type": "array",
                "items": maybe_make_optional(
                    spark_datatype_to_json_schema(dtype.elementType), dtype.containsNull
                ),
            }
        if isinstance(dtype, stypes.BooleanType):
            return {"type": "boolean"}
        if isinstance(dtype, stypes.DoubleType):
            return {"type": "number"}
        if isinstance(dtype, stypes.FloatType):
            return {"type": "number"}
        if isinstance(dtype, stypes.IntegerType):
            return {"type": "integer"}
        if isinstance(dtype, stypes.LongType):
            return {"type": "integer"}
        if isinstance(dtype, stypes.ShortType):
            return {"type": "integer"}
        if isinstance(dtype, stypes.NullType):
            return {"enum": [None]}
        if isinstance(dtype, stypes.StringType):
            return {"type": "string"}

        return {}

    def spark_struct_field_to_json_schema(f: StructField) -> JSON_TYPE:
        type_schema = spark_datatype_to_json_schema(f.dataType)
        result = maybe_make_optional(type_schema, f.nullable)

        if f.name is not None:
            result["description"] = f.name
        return result

    def spark_struct_to_json_schema(
        s: StructType, index_names, table_name: Optional[str] = None
    ) -> JSON_TYPE:
        items = [
            spark_struct_field_to_json_schema(f) for f in s if f.name not in index_names
        ]
        num_items = len(items)
        result = {
            "type": "array",
            "items": {
                "type": "array",
                "description": "rows",
                "minItems": num_items,
                "maxItems": num_items,
                "items": items,
            },
        }

        if table_name is not None:
            result["description"] = table_name

        return result

    return spark_struct_to_json_schema(df.schema, df.index_names, get_table_name(df))


def spark_df_to_schema(df) -> JSON_TYPE:
    result = _spark_df_to_schema(df)
    lale.type_checking.validate_is_schema(result)
    return result


def _to_schema(obj) -> JSON_TYPE:
    result = None
    if obj is None:
        result = {"enum": [None]}
    elif isinstance(obj, ndarray):
        result = _ndarray_to_schema(obj)
    elif isinstance(obj, csr_matrix):
        result = _csr_matrix_to_schema(obj)
    elif isinstance(obj, DataFrame):
        result = _dataframe_to_schema(obj)
    elif isinstance(obj, Series):
        result = _series_to_schema(obj)
    elif Tensor is not None and isinstance(obj, Tensor):
        result = _torch_tensor_to_schema(obj)
    elif is_liac_arff(obj):
        result = _liac_arff_to_schema(obj)
    elif isinstance(obj, list):
        result = _list_tensor_to_schema(obj)
    elif _is_spark_df(obj):
        result = _spark_df_to_schema(obj)
    elif lale.type_checking.is_schema(obj):
        result = obj
        # Does not need to validate again the schema
        return result  # type: ignore
    if result is None:
        raise ValueError(f"to_schema(obj), type {type(obj)}, value {obj}")
    return result


def to_schema(obj) -> JSON_TYPE:
    result = _to_schema(obj)
    lale.type_checking.validate_is_schema(result)
    return result
