# Copyright 2020 IBM Corporation
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

import ast  # see also https://greentreesnakes.readthedocs.io/
import pprint
import typing
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import astunparse

AstLits = (ast.Num, ast.Str, ast.List, ast.Tuple, ast.Set, ast.Dict)
AstLit = Union[ast.Num, ast.Str, ast.List, ast.Tuple, ast.Set, ast.Dict]
AstExprs = (
    *AstLits,
    ast.Name,
    ast.Expr,
    ast.UnaryOp,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
)
AstExpr = Union[
    AstLit,
    ast.Name,
    ast.Expr,
    ast.UnaryOp,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
]


class Expr:
    _expr: AstExpr

    def __init__(self, expr: AstExpr):
        self._expr = expr

    def __bool__(self) -> bool:
        raise TypeError(
            f"Cannot convert expression e1=`{str(self)}` to bool."
            "Instead of `e1 and e2`, try writing `[e1, e2]`."
        )

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __eq__(self, other):
        if isinstance(other, Expr):
            comp = ast.Compare(
                left=self._expr, ops=[ast.Eq()], comparators=[other._expr]
            )
            return Expr(comp)
        else:
            return False

    def __ge__(self, other) -> Union[bool, "Expr"]:
        if isinstance(other, Expr):
            comp = ast.Compare(
                left=self._expr, ops=[ast.GtE()], comparators=[other._expr]
            )
            return Expr(comp)
        else:
            return False

    def __getattr__(self, name: str) -> "Expr":
        attr = ast.Attribute(value=self._expr, attr=name)
        return Expr(attr)

    def __getitem__(self, key: Union[int, str, slice]) -> "Expr":
        key_ast: Union[ast.Index, ast.Slice]
        if isinstance(key, int):
            key_ast = ast.Index(ast.Num(n=key))
        elif isinstance(key, str):
            key_ast = ast.Index(ast.Str(s=key))
        elif isinstance(key, slice):
            key_ast = ast.Slice(key.start, key.stop, key.step)
        else:
            raise TypeError(f"expected int, str, or slice, got {type(key)}")
        subscript = ast.Subscript(value=self._expr, slice=key_ast)
        return Expr(subscript)

    def __str__(self) -> str:
        return astunparse.unparse(self._expr).strip()


def _make_ast_expr(arg: Union[Expr, int, float, str, AstExpr]) -> AstExpr:
    if isinstance(arg, Expr):
        return arg._expr
    elif isinstance(arg, (int, float)):
        return ast.Num(n=arg)
    elif isinstance(arg, str):
        return ast.Str(s=arg)
    else:
        assert isinstance(arg, AstExprs), type(arg)
        return arg


def _make_call_expr(name: str, *args: Union[Expr, AstExpr, int, str]) -> Expr:
    func_ast = ast.Name(id=name)
    args_asts = [_make_ast_expr(arg) for arg in args]
    call_ast = ast.Call(func=func_ast, args=args_asts, keywords=[])
    return Expr(call_ast)


def count(group: Expr) -> Expr:
    return _make_call_expr("count", group)


def day_of_month(subject: Expr, fmt: Optional[str] = None) -> Expr:
    if fmt is None:
        return _make_call_expr("day_of_month", subject)
    return _make_call_expr("day_of_month", subject, fmt)


def day_of_week(subject: Expr, fmt: Optional[str] = None) -> Expr:
    if fmt is None:
        return _make_call_expr("day_of_week", subject)
    return _make_call_expr("day_of_week", subject, fmt)


def day_of_year(subject: Expr, fmt: Optional[str] = None) -> Expr:
    if fmt is None:
        return _make_call_expr("day_of_year", subject)
    return _make_call_expr("day_of_year", subject, fmt)


def distinct_count(group: Expr) -> Expr:
    return _make_call_expr("distinct_count", group)


def hour(subject: Expr, fmt: Optional[str] = None) -> Expr:
    if fmt is None:
        return _make_call_expr("hour", subject)
    return _make_call_expr("hour", subject, fmt)


def item(group: Expr, value: Union[int, str]) -> Expr:
    return _make_call_expr("item", group, value)


def max(group: Expr) -> Expr:
    return _make_call_expr("max", group)


def max_gap_to_cutoff(group: Expr, cutoff: Expr) -> Expr:
    return _make_call_expr("max_gap_to_cutoff", group, cutoff)


def mean(group: Expr) -> Expr:
    return _make_call_expr("mean", group)


def min(group: Expr) -> Expr:
    return _make_call_expr("min", group)


def minute(subject: Expr, fmt: Optional[str] = None) -> Expr:
    if fmt is None:
        return _make_call_expr("minute", subject)
    return _make_call_expr("minute", subject, fmt)


def month(subject: Expr, fmt: Optional[str] = None) -> Expr:
    if fmt is None:
        return _make_call_expr("month", subject)
    return _make_call_expr("month", subject, fmt)


def normalized_count(group: Expr) -> Expr:
    return _make_call_expr("normalized_count", group)


def normalized_sum(group: Expr) -> Expr:
    return _make_call_expr("normalized_sum", group)


def recent(series: Expr, age: int) -> Expr:
    return _make_call_expr("recent", series, age)


def recent_gap_to_cutoff(series: Expr, cutoff: Expr, age: int) -> Expr:
    return _make_call_expr("recent_gap_to_cutoff", series, cutoff, age)


def replace(subject: Expr, old2new: Dict[Any, Any]) -> Expr:
    old2new_str = pprint.pformat(old2new)
    module_ast = ast.parse(old2new_str)
    old2new_ast = typing.cast(ast.Expr, module_ast.body[0])
    return _make_call_expr("replace", old2new_ast)


def sum(group: Expr) -> Expr:
    return _make_call_expr("sum", group)


def trend(series: Expr) -> Expr:
    return _make_call_expr("trend", series)


def variance(group: Expr) -> Expr:
    return _make_call_expr("variance", group)


def window_max(series: Expr, size: int) -> Expr:
    return _make_call_expr("window_max", series, size)


def window_max_trend(series: Expr, size: int) -> Expr:
    return _make_call_expr("window_max_trend", series, size)


def window_mean(series: Expr, size: int) -> Expr:
    return _make_call_expr("window_mean", series, size)


def window_mean_trend(series: Expr, size: int) -> Expr:
    return _make_call_expr("window_mean_trend", series, size)


def window_min(series: Expr, size: int) -> Expr:
    return _make_call_expr("window_min", series, size)


def window_min_trend(series: Expr, size: int) -> Expr:
    return _make_call_expr("window_min_trend", series, size)


def window_variance(series: Expr, size: int) -> Expr:
    return _make_call_expr("window_variance", series, size)


def window_variance_trend(series: Expr, size: int) -> Expr:
    return _make_call_expr("window_variance_trend", series, size)


def string_indexer(subject: Expr) -> Expr:
    return _make_call_expr("string_indexer", subject)


it = Expr(ast.Name(id="it"))
