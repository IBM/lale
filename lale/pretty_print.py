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
import collections
import importlib
import json
import pprint
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

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

_Seq = collections.namedtuple('_Seq', ['src_uid', 'src_jsn', 'dst_uid', 'dst_jsn'])
_Par = collections.namedtuple('_Par', ['s0_uid', 's0_jsn', 's1_uid', 's1_jsn'])
_Graph = collections.namedtuple('_Graph', ['steps', 'preds', 'succs'])

def op_kind(op: Union[JSON_TYPE, _Graph, _Seq, _Par]) -> str:
    if isinstance(op, dict):
        return lale.json_operator.json_op_kind(op)
    elif isinstance(op, _Graph):
        return 'Graph'
    elif isinstance(op, _Seq):
        return 'Seq'
    elif isinstance(op, _Par):
        return 'Par'
    raise TypeError(f'type {type(op)}, op {str(op)}')

def _introduce_structure(pipeline: JSON_TYPE) -> Union[JSON_TYPE, _Graph, _Seq, _Par]:
    assert op_kind(pipeline) == 'Pipeline'
    def make_graph(pipeline: JSON_TYPE) -> _Graph:
        preds: Dict[str, List[str]]
        succs: Dict[str, List[str]]
        if op_kind(pipeline) == 'OperatorChoice':
            steps = {'choice': pipeline}
            preds = {'choice': []}
            succs = {'choice': []}
            return _Graph(steps, preds, succs)
        else:
            steps = pipeline['steps']
            preds = { step: [] for step in steps }
            succs = { step: [] for step in steps }
            for (src, dst) in pipeline['edges']:
                preds[dst].append(src)
                succs[src].append(dst)
            return _Graph(steps, preds, succs)
    def find_seq(graph: _Graph) -> Optional[_Seq]:
        for src in graph.steps:
            if len(graph.succs[src]) == 1:
                dst = graph.succs[src][0]
                if len(graph.preds[dst]) == 1:
                    return _Seq(src, graph.steps[src], dst, graph.steps[dst])
        return None
    def find_par(graph: _Graph) -> Optional[_Par]:
        step_uids = list(graph.steps.keys())
        for i0 in range(len(step_uids)):
            for i1 in range(i0 + 1, len(step_uids)):
                s0, s1 = step_uids[i0], step_uids[i1]
                preds0, preds1 = graph.preds[s0], graph.preds[s1]
                if len(preds0) == len(preds1) and set(preds0) == set(preds1):
                    succs0, succs1 = graph.succs[s0], graph.succs[s1]
                    if len(succs0)==len(succs1) and set(succs0)==set(succs1):
                        return _Par(s0, graph.steps[s0], s1, graph.steps[s1])
        return None
    def replace_seq(old_graph: _Graph, seq: _Seq) -> _Graph:
        result = _Graph({}, {}, {})
        for step in old_graph.steps: #careful to keep topological order
            if step == seq.src_uid:
                result.steps[step] = seq
                result.preds[step] = old_graph.preds[seq.src_uid]
                result.succs[step] = old_graph.succs[seq.dst_uid]
            elif step != seq.dst_uid:
                result.steps[step] = old_graph.steps[step]
                def map_step(step):
                    if step in [seq.src_uid, seq.dst_uid]:
                        return seq.src_uid
                    return step
                result.preds[step] = [
                    map_step(pred) for pred in old_graph.preds[step]]
                result.succs[step] = [
                    map_step(succ) for succ in old_graph.succs[step]]
        return result
    def replace_par(old_graph: _Graph, par: _Par) -> _Graph:
        result = _Graph({}, {}, {})
        for step in old_graph.steps: #careful to keep topological order
            if step == par.s0_uid:
                result.steps[step] = par
                result.preds[step] = old_graph.preds[step]
                result.succs[step] = old_graph.succs[step]
            elif step != par.s1_uid:
                result.steps[step] = old_graph.steps[step]
                result.preds[step] = []
                for pred in old_graph.preds[step]:
                    if pred == par.s0_uid:
                        result.preds[step].append(par.s0_uid)
                    elif pred != par.s1_uid:
                        result.preds[step].append(pred)
                result.succs[step] = []
                for succ in old_graph.succs[step]:
                    if succ == par.s0_uid:
                        result.succs[step].append(par.s0_uid)
                    elif succ != par.s1_uid:
                        result.succs[step].append(succ)
        return result
    def replace_reducibles(graph: _Graph) -> Union[JSON_TYPE, _Graph, _Seq, _Par]:
        progress = True
        while progress:
            seq = find_seq(graph)
            if seq is not None:
                graph = replace_seq(graph, seq)
            par = find_par(graph)
            if par is not None:
                graph = replace_par(graph, par)
            progress = seq is not None or par is not None
        if len(graph.steps) == 1:
            return list(graph.steps.values())[0]
        else:
            return graph
    graph = make_graph(pipeline)
    result = replace_reducibles(graph)
    return result

class _CodeGenState:
    def __init__(self):
        self.imports = []
        self.assigns = []
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

def _pipeline_to_string_rec(graph: Union[JSON_TYPE, _Graph, _Seq, _Par], show_imports: bool, gen: _CodeGenState):
    if op_kind(graph) == 'Pipeline':
        structured = _introduce_structure(cast(JSON_TYPE, graph))
        return _pipeline_to_string_rec(structured, show_imports, gen)
    elif op_kind(graph) == 'Graph':
        steps, preds, succs = cast(_Graph, graph)
        dummy = gen.gensym('step')
        step2name = {}
        for step_uid, step_jsn in steps.items():
            if op_kind(step_jsn) == 'IndividualOp':
                step2name[step_uid] = _pipeline_to_string_rec(step_jsn, show_imports, gen)
            else:
                name = gen.gensym('step')
                expr = _pipeline_to_string_rec(step_jsn, show_imports, gen)
                gen.assigns.append(f'{name} = {expr}')
                step2name[step_uid] = name
        make_pipeline = 'get_pipeline_of_applicable_type'
        gen.imports.append(f'from lale.operators import {make_pipeline}')
        gen.assigns.append(
            'pipeline = {}(\n    steps=[{}],\n    edges=[{}])'.format(
                make_pipeline,
                ', '.join([step2name[step] for step in steps]),
                ', '.join([f'({step2name[src]},{step2name[tgt]})'
                           for src in steps for tgt in succs[src]])))
        return None
    elif op_kind(graph) == 'Seq':
        def parens(op):
            result = _pipeline_to_string_rec(op, show_imports, gen)
            if op_kind(op) == 'Par' or op_kind(op) == 'OperatorChoice':
                return f'({result})'
            return result
        graph = cast(_Seq, graph)
        return f'{parens(graph.src_jsn)} >> {parens(graph.dst_jsn)}'
    elif op_kind(graph) == 'Par':
        def parens(op):
            result = _pipeline_to_string_rec(op, show_imports, gen)
            if op_kind(op) == 'Seq' or op_kind(op) == 'OperatorChoice':
                return f'({result})'
            return result
        graph = cast(_Par, graph)
        return f'{parens(graph.s0_jsn)} & {parens(graph.s1_jsn)}'
    elif op_kind(graph) == 'OperatorChoice':
        def parens(op):
            result = _pipeline_to_string_rec(op, show_imports, gen)
            if op_kind(op) == 'Seq' or op_kind(op) == 'Par':
                return f'({result})'
            return result
        graph = cast(JSON_TYPE, graph)
        printed_steps = [parens(step) for step in graph['steps'].values()]
        return ' | '.join(printed_steps)
    elif op_kind(graph) == 'IndividualOp':
        graph = cast(JSON_TYPE, graph)
        name = graph['label']
        import_stmt, op_expr = _indiv_op_jsn_to_string(graph, True)
        gen.imports.append(import_stmt)
        if re.fullmatch(r'.+\(.+\)', op_expr):
            new_name = gen.gensym(lale.helpers.camelCase_to_snake(name))
            gen.assigns.append(f'{new_name} = {op_expr}')
            return new_name
        else:
            return name
    else:
        assert False, f'unexpected type {type(graph)} of graph {graph}'

def _pipeline_to_string(jsn: JSON_TYPE, show_imports: bool) -> str:
    gen = _CodeGenState()
    expr = _pipeline_to_string_rec(jsn, show_imports, gen)
    if expr:
        gen.assigns.append(f'pipeline = {expr}')
    if show_imports:
        imports_set: Set[str] = set()
        imports_list: List[str] = []
        for imp in gen.imports:
            if imp not in imports_set:
                imports_set |= {imp}
                imports_list.append(imp)
        code = imports_list + gen.assigns
    else:
        code = gen.assigns
    result = '\n'.join(code)
    return result

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

def to_string(arg: Union[JSON_TYPE, 'lale.operators.Operator'], show_imports:bool=True, call_depth:int=1) -> str:
    if lale.helpers.is_schema(arg):
        return schema_to_string(cast(JSON_TYPE, arg))
    elif isinstance(arg, lale.operators.IndividualOp):
        jsn = lale.json_operator.to_json(arg, call_depth=call_depth+1)
        import_stmt, op_expr = _indiv_op_jsn_to_string(jsn, show_imports)
        if import_stmt == '':
            return op_expr
        else:
            return import_stmt + '\npipeline = ' + op_expr
    elif isinstance(arg, lale.operators.BasePipeline):
        jsn = lale.json_operator.to_json(arg, call_depth=call_depth+1)
        return _pipeline_to_string(jsn, show_imports)
    else:
        raise ValueError(f'Unexpected argument type {type(arg)} for {arg}')

def ipython_display(arg: Union[JSON_TYPE, 'lale.operators.Operator'], show_imports:bool=True):
    import IPython.display
    pretty_printed = to_string(arg, show_imports, call_depth=3)
    markdown = IPython.display.Markdown(f'```python\n{pretty_printed}\n```')
    IPython.display.display(markdown)
