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

from typing import Dict, Optional
import os
import logging
import math
import re

from lale.search.search_space import *
from lale.util.Visitor import Visitor
from lale.search import schema2search_space as opt

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

def mk_label(label, counter, useCounter = True):
    if counter is None or not useCounter:
        return label
    else:
        return f"{label}{counter}"

@scope.define
def pgo_sample(pgo, sample):
     return pgo[sample]

class SearchSpaceHPExprVisitor(Visitor):
    @classmethod
    def run(cls, space:SearchSpace, name:str):
        visitor = cls(name)
        space_:Any = space
        return space_.accept(visitor, name)


    def __init__(self, name:str):
        super(SearchSpaceHPExprVisitor, self).__init__()

    def visitSearchSpaceEnum(self, space:SearchSpaceEnum, path:str, counter=None):
        if len(space.vals) == 1:
            return space.vals[0]
        else:
            return hp.choice(mk_label(path, counter), space.vals)

    visitSearchSpaceConstant = visitSearchSpaceEnum
    visitSearchSpaceBool = visitSearchSpaceEnum

    def visitSearchSpaceNumber(self, space:SearchSpaceNumber, path:str, counter=None):
        label = mk_label(path, counter)

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
        p = mk_label(path, num) + "_"
        # mypy does not know about the accept method, since it was
        # added via the VisitorMeta class
        contents:Any = space.contents
        ret = [contents.accept(self, p, counter=x) for x in range(num)]
        return tuple(ret) if space.is_tuple else ret

    def visitSearchSpaceArray(self, space:SearchSpaceArray, path:str, counter=None):
        assert space.maximum >= space.minimum
        p = mk_label(path, counter)
        cp = p + "_"

        if space.minimum == space.maximum:
            return self.array_single_expr_(space, cp, space.minimum)
        else:
            exprs = [self.array_single_expr_(space, cp, x) for x in range(space.minimum, space.maximum+1)]
            res = hp.choice(p, exprs)
            return res

    def visitSearchSpaceList(self, space:SearchSpaceList, path:str, counter=None):
        p = mk_label(path, counter)
        # mypy does not know about the accept method, since it was
        # added via the VisitorMeta class
        contents:List[Any] = space.contents
        ret = [sub.accept(self, p, counter=x) for x,sub in enumerate(contents)]
        return tuple(ret) if space.is_tuple else ret

    def visitSearchSpaceObject(self, space:SearchSpaceObject, path:str, counter=None):
        search_space = {}
        any_path = mk_label(path, counter) + "_" + "combos"
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

class SearchSpaceHPStrVisitor(Visitor):
    pgo_dict:Dict[str, FrequencyDistribution]

    @classmethod
    def run(cls, space:SearchSpace, name:str, counter=None, useCounter=True):
        visitor = cls(name)
        space_:Any = space
        return space_.accept(visitor, name, counter=counter, useCounter=useCounter)


    def __init__(self, name:str):
        super(SearchSpaceHPStrVisitor, self).__init__()
        self.pgo_dict = {}

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
            return f"hp.choice('{mk_label(path, counter, useCounter)}', {vals_str})"

    visitSearchSpaceConstant = visitSearchSpaceEnum
    visitSearchSpaceBool = visitSearchSpaceEnum

    def visitSearchSpaceNumber(self, space:SearchSpaceNumber, path:str, counter=None, useCounter=True):
        label = mk_label(path, counter, useCounter)

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
        p = mk_label(path, num, useCounter=useCounter) + "_"
        ret = "(" if space.is_tuple else "["
        # mypy does not know about the accept method, since it was
        # added via the VisitorMeta class
        contents:Any = space.contents
        ret += ",".join((contents.accept(self, p, counter=x, useCounter=useCounter) for x in range(num)))
        ret += ")" if space.is_tuple else "]"
        return ret
    
    def visitSearchSpaceArray(self, space:SearchSpaceArray, path:str, counter=None, useCounter=True)->str:
        assert space.maximum >= space.minimum
        p = mk_label(path, counter, useCounter=useCounter)
        cp = p + "_"

        if space.minimum == space.maximum:
            return self.array_single_str_(space, cp, space.minimum, useCounter=useCounter)
        else:
            res = "hp.choice(" + p + ", ["
            res += ",".join((self.array_single_str_(space, cp, x, useCounter=useCounter) for x in range(space.minimum, space.maximum+1)))
            res += "])"
            return res

    def visitSearchSpaceList(self, space:SearchSpaceList, path:str, counter=None, useCounter=True)->str:
        p = mk_label(path, counter, useCounter=useCounter)
        ret = "(" if space.is_tuple else "["
        # mypy does not know about the accept method, since it was
        # added via the VisitorMeta class
        contents:List[Any] = space.contents
        ret += ",".join((sub.accept(self, p, counter=x, useCounter=useCounter) for x,sub in enumerate(contents)))
        ret += ")" if space.is_tuple else "]"
        return ret

    # Note that this breaks in bad ways if there is a nested SearchSpaceObject
    # It really relies on there being a top level object and no sub-objects
    def visitSearchSpaceObject(self, space:SearchSpaceObject, path:str, counter=None, useCounter=True):
        s_ret = []
        space_name = "search_space"
        any_name = "valid_hyperparam_combinations"
        any_path = mk_label(path, counter, useCounter=useCounter) + "_" + "combos"
        s_ret.append(space_name + " = {}")
        s_ret.append(f"{space_name}['name'] = {space.longName}")
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
        s_ret.append(f"{any_name} = hp.choice('{any_path}', {str_choices})")
        i = 0
        for k in space.keys:
            s_ret.append(f"{space_name}['{k}'] = {any_name}[{i}]")
            i = i + 1
        
        pgo_decls_str:str = ""
        if self.pgo_dict:
            header = """@scope.define
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
            pgo_decls_str = "\n".join(pgo_decls) + "\n"
    
            return header + "\n" + pgo_decls_str + "\n".join(pgo_decls + s_ret)
        else:
            return s_ret
