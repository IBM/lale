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
from io import StringIO
from typing import Any, Dict, Optional, Union

import astunparse

AstLits = (ast.Num, ast.Str, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Constant)
AstLit = Union[ast.Num, ast.Str, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Constant]
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


# !! WORKAROUND !!
# There is a bug with astunparse and Python 3.8.
# https://github.com/simonpercivall/astunparse/issues/43
# Until it is fixed (which may be never), here is a workaround,
# based on the workaround found in https://github.com/juanlao7/codeclose
class FixUnparser(astunparse.Unparser):
    def _Constant(self, t):
        if not hasattr(t, "kind"):
            setattr(t, "kind", None)

        super()._Constant(t)


# !! WORKAROUND !!
# This method should be called instead of astunparse.unparse
def fixedUnparse(tree):
    v = StringIO()
    FixUnparser(tree, file=v)
    return v.getvalue()


class Expr:
    _expr: AstExpr

    def __init__(self, expr: AstExpr, istrue=None):
        # _istrue variable is used to check the boolean nature of
        # '==' and '!=' operator's results.
        self._expr = expr
        self._istrue = istrue

    def __bool__(self) -> bool:
        if self._istrue is not None:
            return self._istrue
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
            return Expr(comp, istrue=self is other)
        elif other is not None:
            comp = ast.Compare(
                left=self._expr, ops=[ast.Eq()], comparators=[ast.Constant(value=other)]
            )
            return Expr(comp, istrue=False)
        else:
            return False

    def __ge__(self, other):
        if isinstance(other, Expr):
            comp = ast.Compare(
                left=self._expr, ops=[ast.GtE()], comparators=[other._expr]
            )
            return Expr(comp)
        elif other is not None:
            comp = ast.Compare(
                left=self._expr,
                ops=[ast.GtE()],
                comparators=[ast.Constant(value=other)],
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

    def __gt__(self, other):
        if isinstance(other, Expr):
            comp = ast.Compare(
                left=self._expr, ops=[ast.Gt()], comparators=[other._expr]
            )
            return Expr(comp)
        elif other is not None:
            comp = ast.Compare(
                left=self._expr, ops=[ast.Gt()], comparators=[ast.Constant(value=other)]
            )
            return Expr(comp)
        else:
            return False

    def __le__(self, other):
        if isinstance(other, Expr):
            comp = ast.Compare(
                left=self._expr, ops=[ast.LtE()], comparators=[other._expr]
            )
            return Expr(comp)
        elif other is not None:
            comp = ast.Compare(
                left=self._expr,
                ops=[ast.LtE()],
                comparators=[ast.Constant(value=other)],
            )
            return Expr(comp)
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, Expr):
            comp = ast.Compare(
                left=self._expr, ops=[ast.Lt()], comparators=[other._expr]
            )
            return Expr(comp)
        elif other is not None:
            comp = ast.Compare(
                left=self._expr, ops=[ast.Lt()], comparators=[ast.Constant(value=other)]
            )
            return Expr(comp)
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, Expr):
            comp = ast.Compare(
                left=self._expr, ops=[ast.NotEq()], comparators=[other._expr]
            )
            return Expr(comp, istrue=self is other)
        elif other is not None:
            comp = ast.Compare(
                left=self._expr,
                ops=[ast.NotEq()],
                comparators=[ast.Constant(value=other)],
            )
            return Expr(comp, istrue=False)
        else:
            return False

    def __str__(self) -> str:
        result = fixedUnparse(self._expr).strip()
        if isinstance(self._expr, (ast.UnaryOp, ast.BinOp, ast.Compare, ast.BoolOp)):
            if result.startswith("(") and result.endswith(")"):
                result = result[1:-1]
        return result

    def __add__(self, other):
        return _make_binop(ast.Add(), self._expr, other)

    def __sub__(self, other):
        return _make_binop(ast.Sub(), self._expr, other)

    def __mul__(self, other):
        return _make_binop(ast.Mult(), self._expr, other)

    def __truediv__(self, other):
        return _make_binop(ast.Div(), self._expr, other)

    def __floordiv__(self, other):
        return _make_binop(ast.FloorDiv(), self._expr, other)

    def __mod__(self, other):
        return _make_binop(ast.Mod(), self._expr, other)

    def __pow__(self, other):
        return _make_binop(ast.Pow(), self._expr, other)


def _make_binop(op, left, other):
    if isinstance(other, Expr):
        e = ast.BinOp(left=left, op=op, right=other._expr)
        return Expr(e)
    elif other is not None:
        e = ast.BinOp(left=left, op=op, right=ast.Constant(value=other))
        return Expr(e)
    else:
        return False


def _make_ast_expr(arg: Union[None, Expr, int, float, str, AstExpr]) -> AstExpr:
    if arg is None:
        return ast.Constant(value=None)
    elif isinstance(arg, Expr):
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


def collect_set(group: Expr) -> Expr:
    return _make_call_expr("collect_set", group)


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


def replace(
    subject: Expr,
    old2new: Dict[Any, Any],
    handle_unknown="identity",
    unknown_value=None,
) -> Expr:
    old2new_str = pprint.pformat(old2new)
    module_ast = ast.parse(old2new_str)
    old2new_ast = typing.cast(ast.Expr, module_ast.body[0])
    assert handle_unknown in ["identity", "use_encoded_value"]
    return _make_call_expr(
        "replace",
        subject,
        old2new_ast,
        handle_unknown,
        unknown_value,
    )


def identity(subject: Expr) -> Expr:
    return _make_call_expr("identity", subject)


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


def first(group: Expr) -> Expr:
    return _make_call_expr("first", group)


def isnan(column: Expr) -> Expr:
    return _make_call_expr("isnan", column)


def isnotnan(column: Expr) -> Expr:
    return _make_call_expr("isnotnan", column)


def isnull(column: Expr) -> Expr:
    return _make_call_expr("isnull", column)


def isnotnull(column: Expr) -> Expr:
    return _make_call_expr("isnotnull", column)


def asc(column: Union[Expr, str]) -> Expr:
    return _make_call_expr("asc", column)


def desc(column: Union[Expr, str]) -> Expr:
    return _make_call_expr("desc", column)


it = Expr(ast.Name(id="it"))


def _it_column(expr):
    if isinstance(expr, ast.Attribute):
        if _is_ast_name_it(expr.value):
            return expr.attr
        else:
            raise ValueError(
                f"Illegal {fixedUnparse(expr)}. Only the access to `it` is supported"
            )
    elif isinstance(expr, ast.Subscript):
        if _is_ast_name_it(expr.value) and isinstance(expr.slice, ast.Index):
            if isinstance(expr.slice.value, ast.Constant):
                return expr.slice.value.value
            elif isinstance(expr.slice.value, ast.Str):
                return expr.slice.value.s
        else:
            raise ValueError(
                f"Illegal {fixedUnparse(expr)}. Only the access to `it` is supported"
            )
    else:
        raise ValueError(
            f"Illegal {fixedUnparse(expr)}. Only the access to `it` is supported"
        )


def _is_ast_name_it(expr):
    return isinstance(expr, ast.Name) and expr.id == "it"
