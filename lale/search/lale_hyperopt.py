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

from typing import Any, Dict, Iterable, List, Optional
import os
import logging
import math
import re

from lale.search.search_space import *
from lale.util.Visitor import Visitor, accept
from lale.search import schema2search_space as opt
from lale import helpers
from lale.sklearn_compat import make_indexed_name
from lale.operators import Operator

from hyperopt import hp
from hyperopt.pyll import scope

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def search_space_to_hp_expr(space:SearchSpace, name:str):
    return SearchSpaceHPExprVisitor.run(space, name)

def search_space_to_hp_str(space:SearchSpace, name:str)->str:
    return SearchSpaceHPStrVisitor.run(space, name)

def search_space_to_str_for_comparison(space:SearchSpace, name:str)->str:
    return SearchSpaceHPStrVisitor.run(space, name, counter=None, useCounter=False)

def _mk_label(label, counter, useCounter = True):
    if counter is None or not useCounter:
        return label
    else:
        return f"{label}{counter}"

@scope.define
def pgo_sample(pgo, sample):
     return pgo[sample]

@scope.define
def make_nested_hyperopt(space):
    return helpers.NestedHyperoptSpace(space)

class SearchSpaceHPExprVisitor(Visitor):
    names:Dict[str,int]

    @classmethod
    def run(cls, space:SearchSpace, name:str):
        visitor = cls(name)
        space_:Any = space
        return space_.accept(visitor, name)

    def __init__(self, name:str):
        super(SearchSpaceHPExprVisitor, self).__init__()
        self.names = {}

    def get_unique_name(self, name:str)->str:
        if name in self.names:
            counter = self.names[name] + 1
            self.names[name] = counter
            return f"{name}${str(counter)}"
        else:
            self.names[name] = 0
            return name

    def mk_label(self, label, counter, useCounter = True):
        return self.get_unique_name(_mk_label(label, counter, useCounter=useCounter))

    def visitSearchSpaceEnum(self, space:SearchSpaceEnum, path:str, counter=None):
        def as_hp_vals(v):
            # Lists are not "safe" to pass to hyperopt without wrapping
            if isinstance(v, (list,tuple,Operator)):
                return helpers.val_wrapper(v)
            else:
                return v

        if len(space.vals) == 1:
            return as_hp_vals(space.vals[0])
        else:
            return hp.choice(self.mk_label(path, counter), [ as_hp_vals(v) for v in space.vals])

    visitSearchSpaceConstant = visitSearchSpaceEnum
    visitSearchSpaceBool = visitSearchSpaceEnum

    def visitSearchSpaceNumber(self, space:SearchSpaceNumber, path:str, counter=None):
        label = self.mk_label(path, counter)

        if space.pgo is not None:
            return scope.pgo_sample(space.pgo, hp.quniform(label, 0, len(space.pgo)-1, 1))

        dist = "uniform"
        if space.distribution:
            dist = space.distribution

        if space.maximum is None:
            raise ValueError(f"maximum not specified for a number with distribution {dist} for {path}")
        max = space.getInclusiveMax()

        # These distributions need only a maximum
        if dist == "integer":
            if not space.discrete:
                raise ValueError(f"integer distribution specified for a non discrete numeric type for {path}")
            return hp.randint(label, max)

        if space.minimum is None:
            raise ValueError(f"minimum not specified for a number with distribution {dist} for {path}")
        min = space.getInclusiveMin()

        if dist == "uniform":
            if space.discrete:
                return scope.int(hp.quniform(label, min, max, 1))
            else:
                return hp.uniform(label, min, max)
        elif dist == "loguniform":
            # for log distributions, hyperopt requires that we provide the log of the min/max
            if min <= 0:
                raise ValueError(f"minimum of 0 specified with a {dist} distribution for {path}.  This is not allowed; please set it (possibly using minimumForOptimizer) to be positive")
            if min > 0:
                min = math.log(min)
            if max > 0:
                max = math.log(max)
            if space.discrete:
                return scope.int(hp.qloguniform(label, min, max, 1))
            else:
                return hp.loguniform(label, min, max)

        else:
            raise ValueError(f"Unknown distribution type: {dist} for {path}")

    def array_single_expr_(self, space:SearchSpaceArray, path:str, num):
        p = _mk_label(path, num) + "_"
        items:Iterable[SearchSpace] = space.items()
        ret = [accept(sub, self, p, counter=x) for x,sub in enumerate(items)]
        return tuple(ret) if space.is_tuple else ret

    def visitSearchSpaceArray(self, space:SearchSpaceArray, path:str, counter=None):
        assert space.maximum >= space.minimum
        p = _mk_label(path, counter)
        cp = p + "_"

        if space.minimum == space.maximum:
            return self.array_single_expr_(space, cp, space.minimum)
        else:
            exprs = [self.array_single_expr_(space, cp, x) for x in range(space.minimum, space.maximum+1)]
            res = hp.choice(p, exprs)
            return res


    def visitSearchSpaceObject(self, space:SearchSpaceObject, path:str, counter=None):
        search_space = {}
        any_path = self.get_unique_name(_mk_label(path, counter) + "_" + "combos")
        search_space['name']=space.longName

        child_counter = None
        def asexpr(key, e):
            nonlocal child_counter
            if e is None:
                return None
            else:
                ee = e.accept(self, path + "_" + key, counter=child_counter)
                if child_counter is None:
                    child_counter = 1
                else:
                    child_counter = child_counter + 1
                return ee

        def choice_as_tuple_expr(c):
            assert len(space.keys) == len(c)
            ret = [asexpr(space.keys[ind], e) for ind, e in enumerate(c)]
            return ret

        choices = [choice_as_tuple_expr(c) for c in space.choices]

        valid_hyperparam_combinations = hp.choice(any_path, choices)
        i = 0
        for k in space.keys:
            search_space[k] = valid_hyperparam_combinations[i]
            i = i + 1
        return search_space

    def visitSearchSpaceProduct(self, prod:SearchSpaceProduct, path:str, counter=None):
        search_spaces = [accept(space, self, self.get_unique_name(make_indexed_name(name, index))) for name, index, space in prod.get_indexed_spaces()]
        return search_spaces

    def visitSearchSpaceSum(self, sum:SearchSpaceSum, path:str, counter=None):
        if len(sum.sub_spaces) == 1:
            return accept(sum.sub_spaces[0], self, "")
        else:
            unique_name:str = self.get_unique_name("choice")
            search_spaces = hp.choice(unique_name, [{i : accept(m, self, "")} for i, m in enumerate(sum.sub_spaces)])
            return search_spaces

    def visitSearchSpaceOperator(self, op:SearchSpaceOperator, path:str, counter=None):
        return scope.make_nested_hyperopt(accept(op.sub_space, self, path))

    def visitSearchSpaceEmpty(self, op:SearchSpaceEmpty, path:str, counter=None):
        raise NotImplementedError("The hyperopt backend can't compile an empty (sub-) search space")

class SearchSpaceHPStrVisitor(Visitor):
    pgo_dict:Dict[str, FrequencyDistribution]
    names:Dict[str,int]

    pgo_header:Optional[str]
    nested_header:Optional[str]
    decls:str

    @classmethod
    def run(cls, space:SearchSpace, name:str, counter=None, useCounter=True):
        visitor = cls(name)
        space_:Any = space
        ret:str = ""
        body = space_.accept(visitor, name, counter=counter, useCounter=useCounter)
        if visitor.pgo_header is not None:
            ret += visitor.pgo_header
        if visitor.nested_header is not None:
            ret += visitor.nested_header
        if visitor.decls:
            ret += visitor.decls + "\n"
        ret += "return " + body
        return ret


    def get_unique_name(self, name:str)->str:
        if name in self.names:
            counter = self.names[name] + 1
            self.names[name] = counter
            return f"{name}${str(counter)}"
        else:
            self.names[name] = 0
            return name

    def get_unique_variable_name(self, name:str)->str:
        if name in self.names:
            counter = self.names[name] + 1
            self.names[name] = counter
            return f"{name}__{str(counter)}"
        else:
            self.names[name] = 0
            return name

    def mk_label(self, label, counter, useCounter = True):
        return self.get_unique_name(_mk_label(label, counter, useCounter=useCounter))

    def __init__(self, name:str):
        super(SearchSpaceHPStrVisitor, self).__init__()
        self.pgo_dict = {}
        self.names = {}
        self.pgo_header = None
        self.nested_header = None
        self.decls = ""

    def visitSearchSpaceEnum(self, space:SearchSpaceEnum, path:str, counter=None, useCounter=True):

        def val_as_str(v):
            if v is None:
                return "null"
            elif isinstance(v, str):
                return f"'{v}'"
            else:
                return str(v)

        if len(space.vals) == 1:
            return val_as_str(space.vals[0])
        else:
            vals_str = "[" + ", ".join([val_as_str(v) for v in space.vals]) + "]"
            return f"hp.choice('{self.mk_label(path, counter, useCounter)}', {vals_str})"

    visitSearchSpaceConstant = visitSearchSpaceEnum
    visitSearchSpaceBool = visitSearchSpaceEnum

    def visitSearchSpaceNumber(self, space:SearchSpaceNumber, path:str, counter=None, useCounter=True):
        label = self.mk_label(path, counter, useCounter=useCounter)

        if space.pgo is not None:
            self.pgo_dict[label] = space.pgo
            return f"scope.pgo_sample(pgo_{label}, hp.quniform('{label}', {0}, {len(space.pgo)-1}, 1))"


        dist = "uniform"
        if space.distribution:
            dist = space.distribution

        if space.maximum is None:
            raise ValueError(f"maximum not specified for a number with distribution {dist} for {path}")
        max = space.getInclusiveMax()

        # These distributions need only a maximum
        if dist == "integer":
            if not space.discrete:
                raise ValueError(f"integer distribution specified for a non discrete numeric type for {path}")

            return f"hp.randint('{label}', {max})"

        if space.minimum is None:
            raise ValueError(f"minimum not specified for a number with distribution {dist} for {path}")
        min = space.getInclusiveMin()

        if dist == "uniform":
            if space.discrete:
                return f"hp.quniform('{label}', {min}, {max}, 1)"
            else:
                return f"hp.uniform('{label}', {min}, {max})"
        elif dist == "loguniform":
            # for log distributions, hyperopt requires that we provide the log of the min/max
            if min <= 0:
                    raise ValueError(f"minimum of 0 specified with a {dist} distribution for {path}.  This is not allowed; please set it (possibly using minimumForOptimizer) to be positive")
            if min > 0:
                min = math.log(min)
            if max > 0:
                max = math.log(max)

            if space.discrete:
                return f"hp.qloguniform('{label}', {min}, {max}, 1)"
            else:
                return f"hp.loguniform('{label}', {min}, {max})"
        else:
            raise ValueError(f"Unknown distribution type: {dist} for {path}")

    def array_single_str_(self, space:SearchSpaceArray, path:str, num, useCounter=True)->str:
        p = _mk_label(path, num, useCounter=useCounter) + "_"
        ret = "(" if space.is_tuple else "["
        items:Iterable[SearchSpace] = space.items()
        ret += ",".join((accept(sub, self, p, counter=x, useCounter=useCounter) for x,sub in enumerate(items)))
        ret += ")" if space.is_tuple else "]"
        return ret
    
    def visitSearchSpaceArray(self, space:SearchSpaceArray, path:str, counter=None, useCounter=True)->str:
        assert space.maximum >= space.minimum
        p = _mk_label(path, counter, useCounter=useCounter)
        cp = p + "_"

        if space.minimum == space.maximum:
            return self.array_single_str_(space, cp, space.minimum, useCounter=useCounter)
        else:
            res = "hp.choice(" + p + ", ["
            res += ",".join((self.array_single_str_(space, cp, x, useCounter=useCounter) for x in range(space.minimum, space.maximum+1)))
            res += "])"
            return res

    def visitSearchSpaceObject(self, space:SearchSpaceObject, path:str, counter=None, useCounter=True):
        s_decls = []
        space_name = self.get_unique_variable_name("search_space")
        any_name = self.get_unique_variable_name("valid_hyperparam_combinations")
        any_path = self.get_unique_name(_mk_label(path, counter, useCounter=useCounter) + "_" + "combos")
        s_decls.append(space_name + " = {}")
        s_decls.append(f"{space_name}['name'] = {space.longName}")
        child_counter = None

        def cstr(key, x):
            nonlocal child_counter
            if x is None:
                return "None"
            else:
                s = s = x.accept(self, path + "_" + key, child_counter, useCounter=useCounter)
                if child_counter is None:
                    child_counter = 1
                else:
                    child_counter = child_counter + 1
                return s

        def choice_as_tuple_str(c):
            assert len(space.keys) == len(c)
            ret = [cstr(space.keys[ind], e) for ind, e in enumerate(c)]
            return ret

        str_choices = "[" + ",".join(["(" + ",".join(choice_as_tuple_str(c)) + ")" for c in space.choices]) + "]"
        s_decls.append(f"{any_name} = hp.choice('{any_path}', {str_choices})")
        i = 0
        for k in space.keys:
            s_decls.append(f"{space_name}['{k}'] = {any_name}[{i}]")
            i = i + 1
        
        pgo_decls_str:str = ""
        if self.pgo_dict:
            if not self.pgo_header:
                self.pgo_header = """
@scope.define
def pgo_sample(pgo, sample):
    return pgo[sample]

"""

            # use this to sort the pgo_labels by the unique ind
            # appended to the key.
            # This puts them in the order they appear in the hyperopt
            # expression, making it easier to read
            def last_num(kv):
                matches = re.search('(\d+)$', kv[0])
                if matches is None:
                    return 0
                else:
                    return int(matches.group(0))

            pgo_decls:List[str] = []
            for k,v in sorted(self.pgo_dict.items(), key=last_num):
                l = v.freq_dist.tolist()
                pgo_decls.append(f"pgo_{k} = {l}")
            if self.decls:
                self.decls = self.decls + "\n"
            self.decls = self.decls + "\n".join(pgo_decls)
        self.decls += "\n".join(s_decls) + "\n"
        return space_name

    def visitSearchSpaceProduct(self, prod:SearchSpaceProduct, path:str, counter=None, useCounter=True):
        search_spaces = (accept(space, self, self.get_unique_name(make_indexed_name(name, index))) for name, index, space in prod.get_indexed_spaces())
        return "[" + ",".join(search_spaces) + "]"

    def visitSearchSpaceSum(self, sum_space:SearchSpaceSum, path:str, counter=None, useCounter=True):
        unique_name:str = self.get_unique_name("choice")
        sub_str:Iterable[str] = ("\"" + str(i) + "\"" + " : " + "\"" + accept(m, self, "") + "\"" for i, m in enumerate(sum_space.sub_spaces))
        sub_spaces_str:str = "[" + ",".join(sub_str) + "]"
        return f"hp.choice({unique_name}, {sub_spaces_str})"

    def visitSearchSpaceOperator(self, op:SearchSpaceOperator, path:str, counter=None, useCounter=True):
        if not self.nested_header:
            self.nested_header = """
@scope.define
def make_nested_hyperopt(space):
    from lale.helpers import NestedHyperoptSpace
    return NestedHyperoptSpace(space)

"""
        return f"scope.make_nested_hyperopt({accept(op.sub_space, self, path, counter=counter, useCounter=useCounter)})"

    def visitSearchSpaceEmpty(self, op:SearchSpaceEmpty, path:str, counter=None, useCounter=True)->str:
        return "***EMPTY**"
