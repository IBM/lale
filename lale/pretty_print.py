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

import ast
import astunparse
import importlib
import inspect
import json
import pprint
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import lale.helpers
import lale.json_operator
import lale.operators

JSON_TYPE = Dict[str, Any]

def hyperparams_to_string(hps: JSON_TYPE, op:'lale.operators.Operator'=None) -> str:
    if op:
        for k, v in hps.items():
            pass #TODO: use enums where possible
    def value_to_string(value):
        return pprint.pformat(value, width=10000, compact=True)
    strings = [f'{k}={value_to_string(v)}' for k, v in hps.items()]
    return ', '.join(strings)

def _get_module_name(op_label: str, op_name: str, class_name: str) -> str:
    def has_op(module_name, sym):
        module = importlib.import_module(module_name)
        if hasattr(module, sym):
            op = getattr(module, sym)
            if isinstance(op, lale.operators.IndividualOp):
                return op.class_name() == class_name
            else:
                return hasattr(op, '__init__') and hasattr(op, 'fit') and (
                    hasattr(op, 'predict') or hasattr(op, 'transform'))
        return False
    mod_name_long = class_name[:class_name.rfind('.')]
    mod_name_short = mod_name_long[:mod_name_long.rfind('.')]
    unqualified = class_name[class_name.rfind('.')+1:]
    if has_op(mod_name_short, op_label):
        return mod_name_short
    if has_op(mod_name_long, op_label):
        return mod_name_long
    if has_op(mod_name_short, op_name):
        return mod_name_short
    if has_op(mod_name_long, op_name):
        return mod_name_long
    if has_op(mod_name_short, unqualified):
        return mod_name_short
    if has_op(mod_name_long, unqualified):
        return mod_name_long
    assert False, (op_label, op_name, class_name)

def _indiv_op_jsn_to_string(jsn: JSON_TYPE, show_imports: bool) -> Tuple[str, str]:
    assert lale.json_operator.json_op_kind(jsn) == 'IndividualOp'
    label = jsn['label']
    if show_imports:
        class_name = jsn['class']
        module_name = _get_module_name(label, jsn['operator'], class_name)
        if module_name.startswith('lale.'):
            op_name = jsn['operator']
        else:
            op_name = class_name[class_name.rfind('.')+1:]
        if op_name == label:
            import_stmt = f'from {module_name} import {op_name}'
        else:
            import_stmt = f'from {module_name} import {op_name} as {label}'
    else:
        import_stmt = ''
    if 'hyperparams' in jsn and jsn['hyperparams'] is not None:
        hps = hyperparams_to_string(jsn['hyperparams'])
        op_expr = f'{label}({hps})'
    else:
        op_expr = label
    return import_stmt, op_expr

def _indiv_op_to_string(op: 'lale.operators.IndividualOp', show_imports: bool, name:str=None, module_name:str=None) -> Tuple[str, str]:
    assert isinstance(op, lale.operators.IndividualOp)
    if name is None:
        name = op.name()
    if show_imports:
        assert module_name is not None
        if module_name.startswith('lale.'):
            op_name = op.name()
        else:
            op_name = op.class_name().split('.')[-1]
        if name == op_name:
            import_stmt = f'from {module_name} import {op_name}'
        else:
            import_stmt = f'from {module_name} import {op_name} as {name}'
    else:
        import_stmt = ''
    if hasattr(op._impl, "fit") and isinstance(op, lale.operators.TrainableIndividualOp):
        hps = hyperparams_to_string(op.hyperparams(), op)
        op_expr = f'{name}({hps})'
    else:
        op_expr = name
    return import_stmt, op_expr

class _Seq:
    def __init__(self, src, dst):
        self._src = src
        self._dst = dst
    def src(self):
        return self._src
    def dst(self):
        return self._dst
class _Par:
    def __init__(self, s0, s1):
        self._s0 = s0
        self._s1 = s1
    def s0(self):
        return self._s0
    def s1(self):
        return self._s1

def _introduce_structure(pipeline: 'lale.operators.BasePipeline') -> Tuple[List[Any], Dict[Any, List[Any]], Dict[Any, List[Any]]]:
    assert isinstance(pipeline, lale.operators.BasePipeline)
    def shallow_copy_graph(pipeline):
        if isinstance(pipeline, lale.operators.OperatorChoice):
            return [pipeline], {pipeline:[]}, {pipeline:[]}
        steps = [*pipeline.steps()]
        preds = { step: [] for step in steps }
        succs = { step: [] for step in steps }
        for (src, dst) in pipeline.edges():
            preds[dst].append(src)
            succs[src].append(dst)
        return steps, preds, succs
    def find_seq(steps, preds, succs):
        for src in steps:
            if len(succs[src]) == 1:
                dst = succs[src][0]
                if len(preds[dst]) == 1:
                    return _Seq(src, dst)
        return None
    def find_par(steps, preds, succs):
        for i0 in range(len(steps)):
            for i1 in range(i0 + 1, len(steps)):
                s0, s1 = steps[i0], steps[i1]
                preds0, preds1 = preds[s0], preds[s1]
                if len(preds0) == len(preds1) and set(preds0) == set(preds1):
                    succs0, succs1 = succs[s0], succs[s1]
                    if len(succs0)==len(succs1) and set(succs0)==set(succs1):
                        return _Par(s0, s1)
        return None
    def replace_seq(old_steps, old_preds, old_succs, seq):
        new_steps, new_preds, new_succs = [], {}, {}
        for step in old_steps: #careful to keep topological order
            if step is seq.src():
                new_steps.append(seq)
                new_preds[seq] = old_preds[seq.src()]
                new_succs[seq] = old_succs[seq.dst()]
            elif step is not seq.dst():
                new_steps.append(step)
                def map_step(step):
                    if step in [seq.src(), seq.dst()]:
                        return seq
                    return step
                new_preds[step] = [map_step(pred) for pred in old_preds[step]]
                new_succs[step] = [map_step(succ) for succ in old_succs[step]]
        return new_steps, new_preds, new_succs
    def replace_par(old_steps, old_preds, old_succs, par):
        new_steps, new_preds, new_succs = [], {}, {}
        for step in old_steps: #careful to keep topological order
            if step is par.s0():
                new_steps.append(par)
                new_preds[par] = old_preds[step]
                new_succs[par] = old_succs[step]
            elif step is not par.s1():
                new_steps.append(step)
                new_preds[step] = []
                for pred in old_preds[step]:
                    if pred is par.s0():
                        new_preds[step].append(par)
                    elif pred is not par.s1():
                        new_preds[step].append(pred)
                new_succs[step] = []
                for succ in old_succs[step]:
                    if succ is par.s0():
                        new_succs[step].append(par)
                    elif succ is not par.s1():
                        new_succs[step].append(succ)
        return new_steps, new_preds, new_succs
    def replace_reducibles(steps, preds, succs):
        progress = True
        while progress:
            seq = find_seq(steps, preds, succs)
            if seq:
                steps, preds, succs = replace_seq(steps, preds, succs, seq)
            par = find_par(steps, preds, succs)
            if par:
                steps, preds, succs = replace_par(steps, preds, succs, par)
            progress = seq or par
        if len(steps) == 1:
            return steps[0]
        else:
            return steps, preds, succs
    steps, preds, succs = shallow_copy_graph(pipeline)
    graph = replace_reducibles(steps, preds, succs)
    return graph

def _pipeline_to_string(pipeline: 'lale.operators.BasePipeline', show_imports: bool, cls2name: Dict[str,str]) -> str:
    assert isinstance(pipeline, lale.operators.BasePipeline)
    class CodeGenState:
        def __init__(self):
            self.imports = []
            self.assigns = []
            self.irreducibles = []
            self.pipeline = []
            self._names = {'lale','pipeline','get_pipeline_of_applicable_type'}
        def gensym(self, prefix):
            if prefix in self._names:
                suffix = 1
                while f'{prefix}_{suffix}' in self._names:
                    suffix += 1
                result = f'{prefix}_{suffix}'
            else:
                result = prefix
            self._names |= {result}
            return result
    gen = CodeGenState()
    def code_gen_rec(graph):
        if type(graph) is tuple:
            steps, preds, succs = graph
            dummy = gen.gensym('step')
            step2name = {}
            for step in steps:
                if isinstance(step, lale.operators.IndividualOp):
                    step2name[step] = code_gen_rec(step)
                else:
                    name = gen.gensym('step')
                    expr = code_gen_rec(step)
                    gen.irreducibles.append(f'{name} = {expr}')
                    step2name[step] = name
            make_pipeline = 'get_pipeline_of_applicable_type'
            gen.imports.append(f'from lale.operators import {make_pipeline}')
            gen.pipeline = 'pipeline = {}(\n    steps=[{}],\n    edges=[{}])' \
               .format(make_pipeline,
                       ', '.join([step2name[step] for step in steps]),
                       ', '.join([f'({step2name[src]},{step2name[tgt]})'
                                  for src in steps for tgt in succs[src]]))
            return None
        elif isinstance(graph, _Seq):
            def parens(op):
                result = code_gen_rec(op)
                if isinstance(op, _Par) or isinstance(op, lale.operators.OperatorChoice):
                    return f'({result})'
                return result
            return f'{parens(graph.src())} >> {parens(graph.dst())}'
        elif isinstance(graph, _Par):
            def parens(op):
                result = code_gen_rec(op)
                if isinstance(op, _Seq) or isinstance(op, lale.operators.OperatorChoice):
                    return f'({result})'
                return result
            return f'{parens(graph.s0())} & {parens(graph.s1())}'
        elif isinstance(graph, lale.operators.OperatorChoice):
            def parens(op):
                result = code_gen_rec(op)
                if isinstance(op, _Seq) or isinstance(op, _Par):
                    return f'({result})'
                return result
            printed_steps = [parens(step) for step in graph.steps()]
            return ' | '.join(printed_steps)
        elif isinstance(graph, lale.operators.IndividualOp):
            name = gen.gensym(cls2name[graph.class_name()])
            module_name = _get_module_name(graph.name(), graph.name(), graph.class_name())
            import_stmt, op_expr = _indiv_op_to_string(graph, True, name, module_name)
            gen.imports.append(import_stmt)
            if re.fullmatch(r'.+\(.+\)', op_expr):
                new_name = gen.gensym(lale.helpers.camelCase_to_snake(name))
                gen.assigns.append(f'{new_name} = {op_expr}')
                return new_name
            else:
                return name
        else:
            assert False, f'unexpected type {type} of graph {graph}'
    def code_gen_top(graph):
        expr = code_gen_rec(graph)
        if expr:
            gen.pipeline = f'pipeline = {expr}'
        code = gen.imports if show_imports else []
        code = code + gen.assigns + gen.irreducibles + [gen.pipeline]
        result = '\n'.join(code)
        return result
    graph = _introduce_structure(pipeline)
    return code_gen_top(graph)

def schema_to_string(schema: JSON_TYPE) -> str:
    s1 = json.dumps(schema)
    s2 = ast.parse(s1)
    s3 = astunparse.unparse(s2).strip()
    s4 = re.sub(r'}, {\n    (\s+)', r'},\n\1{   ', s3)
    s5 = re.sub(r'\[{\n    (\s+)', r'[\n\1{   ', s4)
    s6 = re.sub(r"'\$schema':[^\n{}\[\]]+\n\s+", "\1", s5)
    while True:
        s7 = re.sub(r',\n\s*([\]}])', r'\1', s6)
        if s6 == s7:
            break
        s6 = s7
    s8 = re.sub(r'{\s+}', r'{}', s7)
    return s8

def to_string(arg: Union[JSON_TYPE, 'lale.operators.Operator'], show_imports:bool=True, call_depth:int=2) -> str:
    def get_cls2name():
        frame = inspect.stack()[call_depth][0]
        result = {}
        all_items = [*frame.f_locals.items(), *frame.f_globals.items()]
        for nm, op in all_items:
            if isinstance(op, lale.operators.IndividualOp) and nm[0].isupper():
                cls = op.class_name()
                if cls not in result:
                    result[cls] = nm
        return result
    if lale.helpers.is_schema(arg):
        return schema_to_string(cast(JSON_TYPE, arg))
    elif isinstance(arg, lale.operators.IndividualOp):
        jsn = lale.json_operator.to_json(arg, call_depth=2)
        import_stmt, op_expr = _indiv_op_jsn_to_string(jsn, show_imports)
        if import_stmt == '':
            return op_expr
        else:
            return import_stmt + '\npipeline = ' + op_expr
    elif isinstance(arg, lale.operators.BasePipeline):
        return _pipeline_to_string(arg, show_imports, get_cls2name())
    else:
        raise ValueError(f'Unexpected argument type {type(arg)} for {arg}')

def ipython_display(arg: Union[JSON_TYPE, 'lale.operators.Operator'], show_imports:bool=True):
    import IPython.display
    pretty_printed = to_string(arg, show_imports, call_depth=3)
    markdown = IPython.display.Markdown(f'```python\n{pretty_printed}\n```')
    IPython.display.display(markdown)
