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

import ast #see also https://greentreesnakes.readthedocs.io/
import astunparse
from typing import Union

class Expr:
    _expr : ast.Expr

    def __init__(self, expr : ast.Expr):
        self._expr = expr

    def __bool__(self) -> bool:
        raise TypeError(f'Cannot convert expression e1=`{str(self)}` to bool.'
                        'Instead of `e1 and e2`, try writing `[e1, e2]`.')

    def __eq__(self, other) -> 'Expr':
        if isinstance(other, Expr):
            comp = ast.Compare(left=self._expr, ops=[ast.Eq()],
                               comparators=[other._expr])
            return Expr(comp)
        else:
            return False

    def __ge__(self, other) -> 'Expr':
        if isinstance(other, Expr):
            comp = ast.Compare(left=self._expr, ops=[ast.GtE()],
                               comparators=[other._expr])
            return Expr(comp)
        else:
            return False

    def __getattr__(self, name : str) -> 'Expr':
        attr = ast.Attribute(value=self._expr, attr=name)
        return Expr(attr)

    def __getitem__(self, key : Union[int, slice]) -> 'Expr':
        if isinstance(key, int):
            key_ast = ast.Index(key)
        elif isinstance(key, slice):
            key_ast = ast.Slice(key.start, key.stop, key.step)
        else:
            raise TypeError(f'expected int or slice, got {key}: {type(key)}')
        subscript = ast.Subscript(value=self._expr, slice=key_ast)
        return Expr(subscript)

    def __str__(self) -> str:
        return astunparse.unparse(self._expr).strip()

def count(group: Expr) -> Expr:
    call = ast.Call(func=ast.Name(id='count'), args=[group._expr], keywords=[])
    return Expr(call)

def max(group: Expr) -> Expr:
    call = ast.Call(func=ast.Name(id='max'), args=[group._expr], keywords=[])
    return Expr(call)

def sum(group: Expr) -> Expr:
    call = ast.Call(func=ast.Name(id='sum'), args=[group._expr], keywords=[])
    return Expr(call)

it = Expr(ast.Name(id='it'))
