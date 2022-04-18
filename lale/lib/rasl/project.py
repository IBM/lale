# Copyright 2019-2022 IBM Corporation
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
from typing import Any, List, Optional

import numpy as np
from typing_extensions import Protocol

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators
import lale.type_checking
from lale.expressions import it
from lale.lib.dataframe import column_index, get_columns
from lale.type_checking import is_schema, validate_is_schema

from .map import Map
from .monoid import Monoid, MonoidFactory


def _columns_schema_to_list(X, schema) -> List[column_index]:
    s_all = lale.datasets.data_schemas._to_schema(X)
    s_row = s_all["items"]
    n_columns = s_row["minItems"]
    assert n_columns == s_row["maxItems"]
    s_cols = s_row["items"]
    if isinstance(s_cols, dict):
        if lale.type_checking.is_subschema(s_cols, schema):
            result = get_columns(X)
        else:
            result = []
    else:
        assert isinstance(s_cols, list)
        cols = get_columns(X)
        result = [
            cols[i]
            for i in range(n_columns)
            if lale.type_checking.is_subschema(s_cols[i], schema)
        ]
    return result


class _ProjectMonoid(Monoid):
    def __init__(self, columns: Monoid, drop_columns: Monoid):
        self._columns = columns
        self._drop_columns = drop_columns

    def combine(self, other: "_ProjectMonoid"):
        c = self._columns.combine(other._columns)
        dc = self._drop_columns.combine(other._drop_columns)
        return _ProjectMonoid(c, dc)

    @property
    def is_absorbing(self):
        return self._columns.is_absorbing and self._drop_columns.is_absorbing


class _StaticMonoid(Monoid):
    def __init__(self, v):
        self._v = v

    def combine(self, other: "_StaticMonoid"):
        assert self._v == other._v
        return self

    @property
    def is_absorbing(self):
        return True


class _StaticMonoidFactory(MonoidFactory[Any, List[column_index], _StaticMonoid]):
    def __init__(self, cl):
        self._cl = cl

    def to_monoid(self, df):
        cl = self._cl
        if cl is None:
            cl = get_columns(df)
            self._cl = cl

        return _StaticMonoid(cl)

    def from_monoid(self, v: _StaticMonoid):
        return v._v


class _DynamicMonoidFactory(
    MonoidFactory[Any, List[column_index], _StaticMonoid], Protocol
):
    pass


class _CallableMonoidFactory(_DynamicMonoidFactory):
    def __init__(self, c):
        self._c = c

    def to_monoid(self, df):
        c = self._c
        if not isinstance(c, list):
            assert callable(c)
            c = c(df)
            self._c = c

        return _StaticMonoid(c)

    def from_monoid(self, v: _StaticMonoid):
        return v._v


class _SchemaMonoidFactory(_DynamicMonoidFactory):
    def __init__(self, c):
        self._c = c

    def to_monoid(self, df):
        c = self._c
        if not isinstance(c, list):
            assert isinstance(c, dict)
            c = _columns_schema_to_list(df, c)
            self._c = c

        return _StaticMonoid(c)

    def from_monoid(self, v: _StaticMonoid):
        return v._v


class _AllDataMonoidFactory(_CallableMonoidFactory):
    """This really needs all the data, and should not be considered a monoid.
    It is used to simplify the code, but does not enable monoidal fitting
    """

    def __init__(self, c):
        super().__init__(c)


def get_column_factory(columns, kind) -> MonoidFactory:
    if columns is None:
        if kind == "passthrough":
            return _StaticMonoidFactory(None)
        else:
            return _StaticMonoidFactory([])
    elif isinstance(columns, list):
        return _StaticMonoidFactory(columns)
    elif isinstance(columns, MonoidFactory):
        return columns
    elif callable(columns):
        return _AllDataMonoidFactory(columns)
    elif isinstance(columns, dict):
        validate_is_schema(columns)
        return _SchemaMonoidFactory(columns)
    else:
        raise TypeError(f"type {type(columns)}, columns {columns}")


class _ProjectImpl:
    def __init__(self, columns=None, drop_columns=None):
        self._columns = get_column_factory(columns, "passthrough")
        self._drop_columns = get_column_factory(drop_columns, "drop")
        self._monoid = None

        self._hyperparams = {"columns": columns, "drop_columns": drop_columns}

    def __getattribute__(self, item):
        # we want to remove fit if a static column is available
        # since it should be considered already trained
        if item == "fit":
            omit_fit = False
            try:
                cols = super().__getattribute__("_columns")
                drop_cols = super().__getattribute__("_drop_columns")
                if isinstance(cols, _StaticMonoidFactory) and isinstance(
                    drop_cols, _StaticMonoidFactory
                ):
                    omit_fit = True
            except AttributeError:
                pass
            if omit_fit:
                raise AttributeError(
                    "fit cannot be called on a Project that has a static columns and drop_columns or has already been fit"
                )
        elif item in ["to_monoid", "from_monoid", "partial_fit"]:
            omit_monoid = False
            try:
                cols = super().__getattribute__("_columns")
                drop_cols = super().__getattribute__("_drop_columns")
                if isinstance(cols, _AllDataMonoidFactory) or isinstance(
                    drop_cols, _AllDataMonoidFactory
                ):
                    omit_monoid = True
            except AttributeError:
                pass
            if omit_monoid:
                raise AttributeError("monoidal operations not available")
        return super().__getattribute__(item)

    def _to_monoid_internal(self, xy):
        df, _ = xy
        col = self._columns.to_monoid(df)
        dcol = self._drop_columns.to_monoid(df)

        return _ProjectMonoid(col, dcol)

    def to_monoid(self, xy):
        return self._to_monoid_internal(xy)

    def _from_monoid_internal(self, pm: _ProjectMonoid):
        col = self._columns.from_monoid(pm._columns)
        dcol = self._drop_columns.from_monoid(pm._drop_columns)

        self._fit_columns = [c for c in col if c not in dcol]

    def from_monoid(self, pm: _ProjectMonoid):
        self._from_monoid_internal(pm)

    _monoid: Optional[_ProjectMonoid]

    def partial_fit(self, X, y=None):
        if self._monoid is None or not self._monoid.is_absorbing:
            lifted = self._to_monoid_internal((X, y))
            if self._monoid is not None:  # not first fit
                lifted = self._monoid.combine(lifted)
            self._from_monoid_internal(lifted)
        return self

    def _fit_internal(self, X, y=None):
        lifted = self._to_monoid_internal((X, y))
        self._from_monoid_internal(lifted)
        return self

    def fit(self, X, y=None):
        return self._fit_internal(X, y)

    def transform(self, X):
        fitcols = getattr(self, "_fit_columns", None)
        if fitcols is None:
            self._fit_internal(X)
            fitcols = getattr(self, "_fit_columns", None)
            assert fitcols is not None

        if isinstance(X, np.ndarray):
            result = X[:, self._fit_columns]  # type: ignore
        else:
            # use the rasl backend
            cols = [
                X.columns[x] if isinstance(x, int) else x for x in self._fit_columns
            ]
            exprs = {c: it[c] for c in cols}
            m = Map(columns=exprs)
            assert isinstance(m, lale.operators.TrainedOperator)
            result = m.transform(X)
        # elif isinstance(X, pd.DataFrame):
        #     if len(self._fit_columns) == 0 or isinstance(self._fit_columns[0], int):
        #         result = X.iloc[:, self._fit_columns]
        #     else:
        #         result = X[self._fit_columns]
        # else:
        #     raise TypeError(f"type {type(X)}")
        s_X = lale.datasets.data_schemas._to_schema(X)
        s_result = self._transform_schema_nocheck(s_X)
        result = lale.datasets.data_schemas.add_schema(result, s_result, recalc=True)
        result = lale.datasets.data_schemas.add_table_name(
            result, lale.datasets.data_schemas.get_table_name(X)
        )
        return result

    def _transform_schema_nocheck(self, s_X):
        if hasattr(self, "_fit_columns"):
            return self._transform_schema_fit_columns(s_X)
        keep_cols = self._columns
        drop_cols = self._drop_columns
        known_keep_cols = False
        known_drop_cols = False
        if keep_cols is None:
            known_keep_cols = True
        elif isinstance(keep_cols, _SchemaMonoidFactory):
            kc = keep_cols._c
            keep_cols = kc
            known_keep_cols = True
        if drop_cols is None:
            known_drop_cols = True
        elif isinstance(drop_cols, _SchemaMonoidFactory):
            dc = drop_cols._c
            drop_cols = dc
            known_drop_cols = True
        if known_keep_cols and known_drop_cols:
            return self._transform_schema_schema(s_X, keep_cols, drop_cols)
        return s_X

    def transform_schema(self, s_X):
        """Used internally by Lale for type-checking downstream operators."""
        if is_schema(s_X):
            return self._transform_schema_nocheck(s_X)
        else:
            X = lale.datasets.data_schemas.add_schema(s_X)
            assert X is not None
            self.fit(X)
            return self._transform_schema_fit_columns(X.json_schema)

    def _transform_schema_fit_columns(self, s_X):
        s_X = lale.datasets.data_schemas._to_schema(s_X)
        s_row = s_X["items"]
        s_cols = s_row["items"]
        n_columns = len(self._fit_columns)
        if isinstance(s_cols, dict):
            s_cols_result = s_cols
        else:
            name2i = {s_cols[i]["description"]: i for i in range(len(s_cols))}
            keep_cols_i = [
                name2i[col] if isinstance(col, str) else col
                for col in self._fit_columns
            ]
            s_cols_result = [s_cols[i] for i in keep_cols_i]
        s_result = {
            **s_X,
            "items": {
                **s_row,
                "minItems": n_columns,
                "maxItems": n_columns,
                "items": s_cols_result,
            },
        }
        return s_result

    def _transform_schema_schema(self, s_X, s_keep, s_drop):
        def is_keeper(column_schema):
            if s_keep is not None:
                if not lale.type_checking.is_subschema(column_schema, s_keep):
                    return False
            if s_drop is not None:
                if lale.type_checking.is_subschema(column_schema, s_drop):
                    return False
            return True

        s_X = lale.datasets.data_schemas._to_schema(s_X)
        s_row = s_X["items"]
        s_cols = s_row["items"]
        if isinstance(s_cols, dict):
            if is_keeper(s_cols):
                s_row_result = s_row
            else:
                s_row_result = {"type": "array", "minItems": 0, "maxItems": 0}
        else:
            assert isinstance(s_cols, list)
            s_cols_result = [s for s in s_cols if is_keeper(s)]
            n_columns = len(s_cols_result)
            s_row_result = {
                "type": "array",
                "minItems": n_columns,
                "maxItems": n_columns,
                "items": s_cols_result,
            }
        return {"type": "array", "items": s_row_result}


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints, if any.",
            "type": "object",
            "additionalProperties": False,
            "required": ["columns", "drop_columns"],
            "relevantToOptimizer": [],
            "properties": {
                "columns": {
                    "description": """The subset of columns to retain.

The supported column specification formats include some of the ones
from scikit-learn's ColumnTransformer_, and in addition, filtering by
using a JSON subschema_ check.

.. _ColumnTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
.. _subschema: https://github.com/IBM/jsonsubschema""",
                    "anyOf": [
                        {
                            "enum": [None],
                            "description": "If not specified, keep all columns.",
                        },
                        {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Multiple columns by index.",
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Multiple Dataframe columns by names.",
                        },
                        {
                            "laleType": "callable",
                            "description": "Callable that is passed the input data X and can return a list of column names or indices.",
                        },
                        {
                            "type": "object",
                            "description": "Keep columns whose schema is a subschema of this JSON schema.",
                        },
                    ],
                    "default": None,
                },
                "drop_columns": {
                    "description": """The subset of columns to remove.

The `drop_columns` argument supports the same formats as `columns`.
If both are specified, keep everything from `columns` that is not
also in `drop_columns`.""",
                    "anyOf": [
                        {
                            "enum": [None],
                            "description": "If not specified, drop no further columns.",
                        },
                        {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Multiple columns by index.",
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Multiple Dataframe columns by names.",
                        },
                        {
                            "laleType": "callable",
                            "description": "Callable that is passed the input data X and can return a list of column names or indices.",
                        },
                        {
                            "type": "object",
                            "description": "Remove columns whose schema is a subschema of this JSON schema.",
                        },
                    ],
                    "default": None,
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {"description": "Target for supervised learning (ignored)."},
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Projection keeps a subset of the columns, like in relational algebra.

Examples
--------
>>> df = pd.DataFrame(data={'A': [1,2], 'B': ['x','y'], 'C': [3,4]})
>>> keep_numbers = Project(columns={'type': 'number'})
>>> keep_numbers.fit(df).transform(df)
NDArrayWithSchema([[1, 3],
                   [2, 4]])
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.project.html",
    "import_from": "lale.lib.rasl",
    "type": "object",
    "tags": {"pre": ["categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


Project = lale.operators.make_operator(_ProjectImpl, _combined_schemas)

lale.docstrings.set_docstrings(Project)
