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
import importlib
import json
import keyword
import logging
import math
import pprint
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import black
import numpy as np

import lale.expressions
import lale.json_operator
import lale.operators
import lale.type_checking

logger = logging.getLogger(__name__)

JSON_TYPE = Dict[str, Any]
_black78 = black.FileMode(line_length=78)


class _CodeGenState:
    imports: List[str]
    assigns: List[str]
    _names: Set[str]

    def __init__(
        self, names: Set[str], combinators: bool, customize_schema: bool, astype: str
    ):
        self.imports = []
        self.assigns = []
        self.combinators = combinators
        self.customize_schema = customize_schema
        self.astype = astype
        self._names = (
            {
                "make_pipeline_graph",
                "lale",
                "make_choice",
                "make_pipeline",
                "make_union",
                "make_union_no_concat",
                "np",
                "pd",
                "pipeline",
            }
            | set(keyword.kwlist)
            | names
        )

    def gensym(self, prefix: str) -> str:
        if prefix in self._names:
            suffix = 0
            while f"{prefix}_{suffix}" in self._names:
                suffix += 1
            result = f"{prefix}_{suffix}"
        else:
            result = prefix
        self._names |= {result}
        return result


def hyperparams_to_string(
    hps: JSON_TYPE, steps: Optional[Dict[str, str]] = None, gen: _CodeGenState = None
) -> str:
    def value_to_string(value):
        if isinstance(value, dict):
            if "$ref" in value and steps is not None:
                step_uid = value["$ref"].split("/")[-1]
                return steps[step_uid]
            else:
                sl = {f"'{k}': {value_to_string(v)}" for k, v in value.items()}
                return "{" + ", ".join(sl) + "}"
        elif isinstance(value, tuple):
            sl = [value_to_string(v) for v in value]
            return "(" + ", ".join(sl) + ")"
        elif isinstance(value, list):
            sl = [value_to_string(v) for v in value]
            return "[" + ", ".join(sl) + "]"
        elif isinstance(value, (int, float)) and math.isnan(value):
            return "float('nan')"
        elif isinstance(value, np.dtype):
            if gen is not None:
                gen.imports.append("import numpy as np")
            return f"np.{value.__repr__()}"
        elif isinstance(value, np.ndarray):
            if gen is not None:
                gen.imports.append("import numpy as np")
            return f"np.{value.__repr__()}"
        elif isinstance(value, np.ufunc):
            if gen is not None:
                gen.imports.append("import numpy as np")
            return f"np.{value.__name__}"  # type: ignore
        elif isinstance(value, lale.expressions.Expr):
            if gen is not None:
                gen.imports.append("from lale.expressions import it")
                for node in ast.walk(value._expr):
                    if isinstance(node, ast.Call):
                        f: Any = node.func
                        gen.imports.append("from lale.expressions import " + f.id)
            return str(value)
        elif hasattr(value, "__module__") and hasattr(value, "__name__"):
            modules = {"numpy": "np", "pandas": "pd"}
            module = modules.get(value.__module__, value.__module__)
            if gen is not None:
                if value.__module__ == module:
                    gen.imports.append(f"import {module}")
                else:
                    gen.imports.append(f"import {value.__module__} as {module}")
            return f"{module}.{value.__name__}"  # type: ignore
        elif hasattr(value, "get_params"):
            module = value.__module__
            if module.startswith("sklearn."):
                i = module.rfind(".")
                if module[i + 1] == "_":
                    module = module[:i]
            if gen is not None:
                gen.imports.append(f"import {module}")
            printed = pprint.pformat(value, width=10000, compact=True)
            compacted = printed.replace("\n", " ")
            return f"{module}.{compacted}"
        else:
            return pprint.pformat(value, width=10000, compact=True)

    strings = [f"{k}={value_to_string(v)}" for k, v in hps.items()]
    return ", ".join(strings)


def _get_module_name(op_label: str, op_name: str, class_name: str) -> str:
    def find_op(module_name, sym):
        module = importlib.import_module(module_name)
        if hasattr(module, sym):
            op = getattr(module, sym)
            if isinstance(op, lale.operators.IndividualOp):
                if op.class_name() == class_name:
                    return op
            elif hasattr(op, "__init__") and hasattr(op, "fit"):
                if hasattr(op, "predict") or hasattr(op, "transform"):
                    return op
        return None

    mod_name_long = class_name[: class_name.rfind(".")]
    if mod_name_long.rfind(".") == -1:
        mod_name_short = mod_name_long
    else:
        mod_name_short = mod_name_long[: mod_name_long.rfind(".")]
    unqualified = class_name[class_name.rfind(".") + 1 :]
    if (
        class_name.startswith("lale.")
        and unqualified.startswith("_")
        and unqualified.endswith("Impl")
    ):
        unqualified = unqualified[1 : -len("Impl")]
    op = find_op(mod_name_short, op_name)
    if op is not None:
        mod = mod_name_short
    else:
        op = find_op(mod_name_long, op_name)
        if op is not None:
            mod = mod_name_long
        else:
            op = find_op(mod_name_short, unqualified)
            if op is not None:
                mod = mod_name_short
            else:
                op = find_op(mod_name_long, unqualified)
                if op is not None:
                    mod = mod_name_long
                else:
                    assert False, (op_label, op_name, class_name)

    assert op is not None, (op_label, op_name, class_name)
    if isinstance(op, lale.operators.IndividualOp):
        if "import_from" in op._schemas:
            mod = op._schemas["import_from"]
    return mod


def _op_kind(op: JSON_TYPE) -> str:
    assert isinstance(op, dict)
    if "kind" in op:
        return op["kind"]
    return lale.json_operator.json_op_kind(op)


_OP_KIND_TO_COMBINATOR = {"Seq": ">>", "Par": "&", "OperatorChoice": "|"}
_OP_KIND_TO_FUNCTION = {
    "Seq": "make_pipeline",
    "Par": "make_union_no_concat",
    "OperatorChoice": "make_choice",
    "Union": "make_union",
}


def _introduce_structure(pipeline: JSON_TYPE, gen: _CodeGenState) -> JSON_TYPE:
    assert _op_kind(pipeline) == "Pipeline"

    def make_graph(pipeline: JSON_TYPE) -> JSON_TYPE:
        steps = pipeline["steps"]
        preds: Dict[str, List[str]] = {step: [] for step in steps}
        succs: Dict[str, List[str]] = {step: [] for step in steps}
        for (src, dst) in pipeline["edges"]:
            preds[dst].append(src)
            succs[src].append(dst)
        return {"kind": "Graph", "steps": steps, "preds": preds, "succs": succs}

    def find_seq(
        graph: JSON_TYPE,
    ) -> Optional[Tuple[Dict[str, JSON_TYPE], Dict[str, JSON_TYPE]]]:
        for src in graph["steps"]:
            if len(graph["succs"][src]) == 1:
                dst = graph["succs"][src][0]
                if len(graph["preds"][dst]) == 1:
                    old: Dict[str, JSON_TYPE] = {
                        uid: graph["steps"][uid] for uid in [src, dst]
                    }
                    new_uid = None
                    new_steps: Dict[str, JSON_TYPE] = {}
                    for step_uid, step_jsn in old.items():
                        if _op_kind(step_jsn) == "Seq":  # flatten
                            new_steps.update(step_jsn["steps"])
                            if new_uid is None:
                                new_uid = step_uid
                        else:
                            new_steps[step_uid] = step_jsn
                    if new_uid is None:
                        new_uid = gen.gensym("pipeline")
                    new = {new_uid: {"kind": "Seq", "steps": new_steps}}
                    return old, new
        return None

    def find_par(
        graph: JSON_TYPE,
    ) -> Optional[Tuple[Dict[str, JSON_TYPE], Dict[str, JSON_TYPE]]]:
        step_uids = list(graph["steps"].keys())
        for i0 in range(len(step_uids)):
            for i1 in range(i0 + 1, len(step_uids)):
                s0, s1 = step_uids[i0], step_uids[i1]
                preds0, preds1 = graph["preds"][s0], graph["preds"][s1]
                if len(preds0) == len(preds1) and set(preds0) == set(preds1):
                    succs0, succs1 = graph["succs"][s0], graph["succs"][s1]
                    if len(succs0) == len(succs1) and set(succs0) == set(succs1):
                        old: Dict[str, JSON_TYPE] = {
                            uid: graph["steps"][uid] for uid in [s0, s1]
                        }
                        new_uid = None
                        new_steps: Dict[str, JSON_TYPE] = {}
                        for step_uid, step_jsn in old.items():
                            if _op_kind(step_jsn) == "Par":  # flatten
                                new_steps.update(step_jsn["steps"])
                                if new_uid is None:
                                    new_uid = step_uid
                            else:
                                new_steps[step_uid] = step_jsn
                        if new_uid is None:
                            new_uid = gen.gensym("union")
                        new: Dict[str, JSON_TYPE] = {
                            new_uid: {"kind": "Par", "steps": new_steps}
                        }
                        return old, new
        return None

    def find_union(
        graph: JSON_TYPE,
    ) -> Optional[Tuple[Dict[str, JSON_TYPE], Dict[str, JSON_TYPE]]]:
        cat_cls = "lale.lib.lale.concat_features._ConcatFeaturesImpl"
        for seq_uid, seq_jsn in graph["steps"].items():
            if _op_kind(seq_jsn) == "Seq":
                seq_uids = list(seq_jsn["steps"].keys())
                for i in range(len(seq_uids) - 1):
                    src, dst = seq_uids[i], seq_uids[i + 1]
                    src_jsn = seq_jsn["steps"][src]
                    if _op_kind(src_jsn) == "Par":
                        dst_jsn = seq_jsn["steps"][dst]
                        if dst_jsn.get("class", None) == cat_cls:
                            old = {seq_uid: seq_jsn}
                            union = {"kind": "Union", "steps": src_jsn["steps"]}
                            if len(seq_uids) == 2:
                                new = {src: union}
                            else:
                                new_steps: Dict[str, JSON_TYPE] = {}
                                for uid, jsn in seq_jsn["steps"].items():
                                    if uid == src:
                                        new_steps[uid] = union
                                    elif uid != dst:
                                        new_steps[uid] = jsn
                                new = {src: {"kind": "Seq", "steps": new_steps}}
                            return old, new
        return None

    def replace(
        subject: JSON_TYPE, old: Dict[str, JSON_TYPE], new: Dict[str, JSON_TYPE]
    ) -> JSON_TYPE:
        assert _op_kind(subject) == "Graph"
        new_uid, new_jsn = list(new.items())[0]
        assert _op_kind(new_jsn) in ["Seq", "Par", "Union"]
        subj_steps = subject["steps"]
        subj_preds = subject["preds"]
        subj_succs = subject["succs"]
        res_steps: Dict[str, JSON_TYPE] = {}
        res_preds: Dict[str, List[str]] = {}
        res_succs: Dict[str, List[str]] = {}
        old_steps_uids = list(old.keys())
        for step_uid in subj_steps:  # careful to keep topological order
            if step_uid == old_steps_uids[0]:
                res_steps[new_uid] = new_jsn
                res_preds[new_uid] = subj_preds[old_steps_uids[0]]
                res_succs[new_uid] = subj_succs[old_steps_uids[-1]]
            elif step_uid not in old_steps_uids:
                res_steps[step_uid] = subj_steps[step_uid]
                res_preds[step_uid] = []
                for pred in subj_preds[step_uid]:
                    if pred == old_steps_uids[-1]:
                        res_preds[step_uid].append(new_uid)
                    elif pred not in old_steps_uids:
                        res_preds[step_uid].append(pred)
                res_succs[step_uid] = []
                for succ in subj_succs[step_uid]:
                    if succ == old_steps_uids[0]:
                        res_succs[step_uid].append(new_uid)
                    elif succ not in old_steps_uids:
                        res_succs[step_uid].append(succ)
        result = {
            "kind": "Graph",
            "steps": res_steps,
            "preds": res_preds,
            "succs": res_succs,
        }
        return result

    def find_and_replace(graph: JSON_TYPE) -> JSON_TYPE:
        progress = True
        while progress:
            seq = find_seq(graph)
            if seq is not None:
                graph = replace(graph, *seq)
            par = find_par(graph)
            if par is not None:
                graph = replace(graph, *par)
            if not gen.combinators:
                union = find_union(graph)
                if union is not None:
                    graph = replace(graph, *union)
            progress = seq is not None or par is not None
        if len(graph["steps"]) == 1:  # flatten
            return list(graph["steps"].values())[0]
        else:
            return graph

    graph = make_graph(pipeline)
    result = find_and_replace(graph)
    return result


def _operator_jsn_to_string_rec(uid: str, jsn: JSON_TYPE, gen: _CodeGenState) -> str:
    op_expr: str
    if _op_kind(jsn) == "Pipeline":
        structured = _introduce_structure(jsn, gen)
        return _operator_jsn_to_string_rec(uid, structured, gen)
    elif _op_kind(jsn) == "Graph":
        steps, succs = jsn["steps"], jsn["succs"]
        step2name: Dict[str, str] = {}
        for step_uid, step_val in steps.items():
            expr = _operator_jsn_to_string_rec(step_uid, step_val, gen)
            if re.fullmatch("[A-Za-z][A-Za-z0-9_]*", expr):
                step2name[step_uid] = expr
            else:
                step2name[step_uid] = step_uid
                gen.assigns.append(f"{step_uid} = {expr}")
        make_pipeline = "make_pipeline_graph"
        gen.imports.append(f"from lale.operators import {make_pipeline}")
        result = "{}(steps=[{}], edges=[{}])".format(
            make_pipeline,
            ", ".join([step2name[step] for step in steps]),
            ", ".join(
                [
                    f"({step2name[src]},{step2name[tgt]})"
                    for src in steps
                    for tgt in succs[src]
                ]
            ),
        )
        return result
    elif _op_kind(jsn) in ["Seq", "Par", "OperatorChoice", "Union"]:
        if gen.combinators:

            def print_for_comb(step_uid, step_val):
                printed = _operator_jsn_to_string_rec(step_uid, step_val, gen)
                parens = _op_kind(step_val) != _op_kind(jsn) and _op_kind(step_val) in [
                    "Seq",
                    "Par",
                    "OperatorChoice",
                ]
                return f"({printed})" if parens else printed

            printed_steps = {
                step_uid: print_for_comb(step_uid, step_val)
                for step_uid, step_val in jsn["steps"].items()
            }
            combinator = _OP_KIND_TO_COMBINATOR[_op_kind(jsn)]
            return f" {combinator} ".join(printed_steps.values())
        else:
            printed_steps = {
                step_uid: _operator_jsn_to_string_rec(step_uid, step_val, gen)
                for step_uid, step_val in jsn["steps"].items()
            }
            function = _OP_KIND_TO_FUNCTION[_op_kind(jsn)]
            if gen.astype == "sklearn" and function in ["make_union", "make_pipeline"]:
                gen.imports.append(f"from sklearn.pipeline import {function}")
            else:
                gen.imports.append(f"from lale.operators import {function}")
            op_expr = "{}({})".format(function, ", ".join(printed_steps.values()))
            gen.assigns.append(f"{uid} = {op_expr}")
            return uid
    elif _op_kind(jsn) == "IndividualOp":
        label: str = jsn["label"]
        class_name = jsn["class"]
        module_name = _get_module_name(label, jsn["operator"], class_name)
        if module_name.startswith("lale."):
            op_name = jsn["operator"]
        else:
            op_name = class_name[class_name.rfind(".") + 1 :]
        if op_name.startswith("_"):
            op_name = op_name[1:]
        if op_name.endswith("Impl"):
            op_name = op_name[: -len("Impl")]
        if op_name == label:
            import_stmt = f"from {module_name} import {op_name}"
        else:
            import_stmt = f"from {module_name} import {op_name} as {label}"
        gen.imports.append(import_stmt)
        printed_steps = {
            step_uid: _operator_jsn_to_string_rec(step_uid, step_val, gen)
            for step_uid, step_val in jsn.get("steps", {}).items()
        }
        op_expr = label
        if "customize_schema" in jsn and gen.customize_schema:
            if jsn["customize_schema"] == "not_available":
                logger.warning(f"missing {label}.customize_schema(..) call")
            elif jsn["customize_schema"] != {}:
                new_hps = jsn["customize_schema"]["properties"]["hyperparams"]["allOf"][
                    0
                ]
                customize_schema_string = ",".join(
                    [
                        f"{hp_name}={json_to_string(hp_schema)}"
                        for hp_name, hp_schema in new_hps.items()
                    ]
                )
                op_expr = f"{op_expr}.customize_schema({customize_schema_string})"
        if "hyperparams" in jsn and jsn["hyperparams"] is not None:
            hp_string = hyperparams_to_string(jsn["hyperparams"], printed_steps, gen)
            op_expr = f"{op_expr}({hp_string})"
        if re.fullmatch(r".+\(.+\)", op_expr):
            gen.assigns.append(f"{uid} = {op_expr}")
            return uid
        else:
            return op_expr
    else:
        assert False, f"unexpected type {type(jsn)} of jsn {jsn}"


def _collect_names(jsn: JSON_TYPE) -> Set[str]:
    result: Set[str] = set()
    if "steps" in jsn:
        steps: Dict[str, JSON_TYPE] = jsn["steps"]
        for step_uid, step_jsn in steps.items():
            result |= {step_uid}
            result |= _collect_names(step_jsn)
    if "label" in jsn:
        lbl: str = jsn["label"]
        result |= {lbl}
    return result


def _combine_lonely_literals(printed_code):
    lines = printed_code.split("\n")
    regex = re.compile(
        r' +("[^"]*"|\d+\.?\d*|\[\]|float\("nan"\)|np\.dtype\("[^"]+"\)),'
    )
    for i in range(len(lines)):
        if lines[i] is not None:
            match_i = regex.fullmatch(lines[i])
            if match_i is not None:
                j = i + 1
                while j < len(lines) and lines[j] is not None:
                    match_j = regex.fullmatch(lines[j])
                    if match_j is None:
                        break
                    candidate = lines[i] + " " + match_j.group(1) + ","
                    if len(candidate) > 78:
                        break
                    lines[i] = candidate
                    lines[j] = None
                    j += 1
    result = "\n".join([s for s in lines if s is not None])
    return result


def _format_code(printed_code):
    formatted = black.format_str(printed_code, mode=_black78).rstrip()
    combined = _combine_lonely_literals(formatted)
    return combined


def _operator_jsn_to_string(
    jsn: JSON_TYPE,
    show_imports: bool,
    combinators: bool,
    customize_schema: bool,
    astype: str,
) -> str:
    gen = _CodeGenState(_collect_names(jsn), combinators, customize_schema, astype)
    expr = _operator_jsn_to_string_rec("pipeline", jsn, gen)
    if expr != "pipeline":
        gen.assigns.append(f"pipeline = {expr}")
    if show_imports and len(gen.imports) > 0:
        if combinators:
            gen.imports.append("import lale")
        imports_set: Set[str] = set()
        imports_list: List[str] = []
        for imp in gen.imports:
            if imp not in imports_set:
                imports_set |= {imp}
                imports_list.append(imp)
        result = "\n".join(imports_list)
        if combinators:
            result += "\nlale.wrap_imported_operators()"
        result += "\n"
        result += "\n".join(gen.assigns)
    else:
        result = "\n".join(gen.assigns)
    formatted = _format_code(result)
    return formatted


def json_to_string(schema: JSON_TYPE) -> str:
    s1 = json.dumps(schema)
    s2 = _format_code(s1)
    return s2


def to_string(
    arg: Union[JSON_TYPE, "lale.operators.Operator"],
    show_imports: bool = True,
    combinators: bool = True,
    customize_schema: bool = False,
    astype: str = "lale",
    call_depth: int = 1,
) -> str:
    assert astype in ["lale", "sklearn"], astype
    if astype == "sklearn":
        combinators = False
    if lale.type_checking.is_schema(arg):
        return json_to_string(cast(JSON_TYPE, arg))
    elif isinstance(arg, lale.operators.Operator):
        jsn = lale.json_operator.to_json(arg, call_depth=call_depth + 1)
        return _operator_jsn_to_string(
            jsn, show_imports, combinators, customize_schema, astype
        )
    else:
        raise ValueError(f"Unexpected argument type {type(arg)} for {arg}")


def ipython_display(
    arg: Union[JSON_TYPE, "lale.operators.Operator"],
    show_imports: bool = True,
    combinators: bool = True,
):
    import IPython.display

    pretty_printed = to_string(arg, show_imports, combinators, call_depth=3)
    markdown = IPython.display.Markdown(f"```python\n{pretty_printed}\n```")
    IPython.display.display(markdown)
