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

import importlib
import inspect
import keyword
import logging
import re
from typing import Any, Dict, Tuple, cast

import jsonschema

import lale.operators
from lale.helpers import GenSym

logger = logging.getLogger(__name__)

JSON_TYPE = Dict[str, Any]

SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "definitions": {
        "operator": {
            "anyOf": [
                {"$ref": "#/definitions/planned_individual_op"},
                {"$ref": "#/definitions/trainable_individual_op"},
                {"$ref": "#/definitions/trained_individual_op"},
                {"$ref": "#/definitions/planned_pipeline"},
                {"$ref": "#/definitions/trainable_pipeline"},
                {"$ref": "#/definitions/trained_pipeline"},
                {"$ref": "#/definitions/operator_choice"},
            ]
        },
        "individual_op": {
            "type": "object",
            "required": ["class", "state", "operator"],
            "properties": {
                "class": {
                    "type": "string",
                    "pattern": "^([A-Za-z_][A-Za-z_0-9]*[.])*[A-Za-z_][A-Za-z_0-9]*$",
                },
                "state": {"enum": ["metamodel", "planned", "trainable", "trained"]},
                "operator": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z_0-9]*$"},
                "label": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z_0-9]*$"},
                "documentation_url": {"type": "string"},
                "hyperparams": {
                    "anyOf": [
                        {"enum": [None]},
                        {
                            "type": "object",
                            "patternProperties": {"^[A-Za-z_][A-Za-z_0-9]*$": {}},
                        },
                    ]
                },
                "steps": {
                    "description": "Nested operators in higher-order individual op.",
                    "type": "object",
                    "patternProperties": {
                        "^[a-z][a-z_0-9]*$": {"$ref": "#/definitions/operator"}
                    },
                },
                "is_frozen_trainable": {"type": "boolean"},
                "is_frozen_trained": {"type": "boolean"},
                "coefs": {"enum": [None, "coefs_not_available"]},
                "customize_schema": {
                    "anyOf": [
                        {"enum": ["not_available"]},
                        {"type": "object"},
                    ],
                },
            },
        },
        "planned_individual_op": {
            "allOf": [
                {"$ref": "#/definitions/individual_op"},
                {"type": "object", "properties": {"state": {"enum": ["planned"]}}},
            ]
        },
        "trainable_individual_op": {
            "allOf": [
                {"$ref": "#/definitions/individual_op"},
                {
                    "type": "object",
                    "required": ["hyperparams", "is_frozen_trainable"],
                    "properties": {"state": {"enum": ["trainable"]}},
                },
            ]
        },
        "trained_individual_op": {
            "allOf": [
                {"$ref": "#/definitions/individual_op"},
                {
                    "type": "object",
                    "required": ["hyperparams", "coefs", "is_frozen_trained"],
                    "properties": {"state": {"enum": ["trained"]}},
                },
            ]
        },
        "pipeline": {
            "type": "object",
            "required": ["class", "state", "edges", "steps"],
            "properties": {
                "class": {
                    "enum": [
                        "lale.operators.PlannedPipeline",
                        "lale.operators.TrainablePipeline",
                        "lale.operators.TrainedPipeline",
                    ]
                },
                "state": {"enum": ["planned", "trainable", "trained"]},
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "string", "pattern": "^[a-z][a-z_0-9]*$"},
                    },
                },
                "steps": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-z][a-z_0-9]*$": {"$ref": "#/definitions/operator"}
                    },
                },
            },
        },
        "planned_pipeline": {
            "allOf": [
                {"$ref": "#/definitions/pipeline"},
                {
                    "type": "object",
                    "properties": {
                        "state": {"enum": ["planned"]},
                        "class": {"enum": ["lale.operators.PlannedPipeline"]},
                    },
                },
            ]
        },
        "trainable_pipeline": {
            "allOf": [
                {"$ref": "#/definitions/pipeline"},
                {
                    "type": "object",
                    "properties": {
                        "state": {"enum": ["trainable"]},
                        "class": {"enum": ["lale.operators.TrainablePipeline"]},
                        "steps": {
                            "type": "object",
                            "patternProperties": {
                                "^[a-z][a-z_0-9]*$": {
                                    "type": "object",
                                    "properties": {
                                        "state": {"enum": ["trainable", "trained"]}
                                    },
                                }
                            },
                        },
                    },
                },
            ]
        },
        "trained_pipeline": {
            "allOf": [
                {"$ref": "#/definitions/pipeline"},
                {
                    "type": "object",
                    "properties": {
                        "state": {"enum": ["trained"]},
                        "class": {"enum": ["lale.operators.TrainedPipeline"]},
                        "steps": {
                            "type": "object",
                            "patternProperties": {
                                "^[a-z][a-z_0-9]*$": {
                                    "type": "object",
                                    "properties": {"state": {"enum": ["trained"]}},
                                }
                            },
                        },
                    },
                },
            ]
        },
        "operator_choice": {
            "type": "object",
            "required": ["class", "state", "operator", "steps"],
            "properties": {
                "class": {"enum": ["lale.operators.OperatorChoice"]},
                "state": {"enum": ["planned"]},
                "operator": {"type": "string"},
                "steps": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-z][a-z_0-9]*$": {"$ref": "#/definitions/operator"}
                    },
                },
            },
        },
    },
    "$ref": "#/definitions/operator",
}


def json_op_kind(jsn: JSON_TYPE) -> str:
    if jsn["class"] == "lale.operators.OperatorChoice":
        return "OperatorChoice"
    if jsn["class"] in [
        "lale.operators.PlannedPipeline",
        "lale.operators.TrainablePipeline",
        "lale.operators.TrainedPipeline",
    ]:
        return "Pipeline"
    return "IndividualOp"


def _get_state(op: "lale.operators.Operator") -> str:
    if isinstance(op, lale.operators.TrainedOperator):
        return "trained"
    if isinstance(op, lale.operators.TrainableOperator):
        return "trainable"
    if isinstance(op, lale.operators.PlannedOperator) or isinstance(
        op, lale.operators.OperatorChoice
    ):
        return "planned"
    if isinstance(op, lale.operators.Operator):
        return "metamodel"
    raise TypeError(f"Expected lale.operators.Operator, got {type(op)}.")


def _get_cls2label(call_depth: int) -> Dict[str, str]:
    inspect_stack = inspect.stack()
    if call_depth >= len(inspect_stack):
        return {}
    frame = inspect_stack[call_depth][0]
    cls2label: Dict[str, str] = {}
    cls2state: Dict[str, str] = {}
    all_items: Dict[str, Any] = {**frame.f_locals, **frame.f_globals}
    for label, op in all_items.items():
        if isinstance(op, lale.operators.IndividualOp):
            state = _get_state(op)
            cls = op.class_name()
            if cls in cls2state:
                insert = (
                    (cls2state[cls] == "trainable" and state == "planned")
                    or (
                        cls2state[cls] == "trained"
                        and state in ["trainable", "planned"]
                    )
                    or (cls2state[cls] == state and label[0].isupper())
                )
            else:
                insert = True
            if insert:
                if not label.islower():
                    cls2label[cls] = label
                    cls2state[cls] = state
    return cls2label


def _camelCase_to_snake(name):
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _init_gensym(op: "lale.operators.Operator", cls2label: Dict[str, str]) -> GenSym:
    label2count: Dict[str, int] = {}

    def populate_label2count(op: "lale.operators.Operator"):
        if isinstance(op, lale.operators.IndividualOp):
            label = cls2label.get(op.class_name(), op.name())
        elif isinstance(op, lale.operators.BasePipeline):
            for s in op.steps_list():
                populate_label2count(s)
            label = "pipeline"
        elif isinstance(op, lale.operators.OperatorChoice):
            for s in op.steps_list():
                populate_label2count(s)
            label = "choice"
        else:
            raise ValueError(f"Unexpected argument of type: {type(op)}")
        label2count[label] = label2count.get(label, 0) + 1

    populate_label2count(op)
    non_unique_labels = {ll for ll, c in label2count.items() if c > 1}
    snakes = {_camelCase_to_snake(ll) for ll in non_unique_labels}
    return GenSym(
        {"lale", "make_pipeline", "make_union", "make_choice"}
        | set(keyword.kwlist)
        | non_unique_labels
        | snakes
    )


def _hps_to_json_rec(
    hps,
    cls2label: Dict[str, str],
    gensym: GenSym,
    steps,
    add_custom_default: bool,
) -> Any:
    if isinstance(hps, lale.operators.Operator):
        step_uid, step_jsn = _op_to_json_rec(hps, cls2label, gensym, add_custom_default)
        steps[step_uid] = step_jsn
        return {"$ref": f"../steps/{step_uid}"}
    elif isinstance(hps, dict):
        return {
            hp_name: _hps_to_json_rec(
                hp_val, cls2label, gensym, steps, add_custom_default
            )
            for hp_name, hp_val in hps.items()
        }
    elif isinstance(hps, tuple):
        return tuple(
            [
                _hps_to_json_rec(hp_val, cls2label, gensym, steps, add_custom_default)
                for hp_val in hps
            ]
        )
    elif isinstance(hps, list):
        return [
            _hps_to_json_rec(hp_val, cls2label, gensym, steps, add_custom_default)
            for hp_val in hps
        ]
    else:
        return hps


def _get_customize_schema(after, before):
    if after == before:
        return {}
    if after is None or before is None:
        return "not_available"

    def dict_equal_modulo(d1, d2, mod):
        for k in d1.keys():
            if k != mod and (k not in d2 or d1[k] != d2[k]):
                return False
        for k in d2.keys():
            if k != mod and k not in d1:
                return False
        return True

    def list_equal_modulo(l1, l2, mod):
        if len(l1) != len(l2):
            return False
        for i in range(len(l1)):
            if i != mod and l1[i] != l2[i]:
                return False
        return True

    if not dict_equal_modulo(after, before, "properties"):
        return "not_available"
    after = after["properties"]
    before = before["properties"]
    if not dict_equal_modulo(after, before, "hyperparams"):
        return "not_available"
    after = after["hyperparams"]["allOf"]
    before = before["hyperparams"]["allOf"]
    if not list_equal_modulo(after, before, 0):
        return "not_available"
    after = after[0]
    before = before[0]
    if not dict_equal_modulo(after, before, "properties"):
        return "not_available"
    after = after["properties"]
    before = before["properties"]
    # TODO: only supports customizing the schema for individual hyperparams
    hp_diff = {
        hp_name: hp_schema
        for hp_name, hp_schema in after.items()
        if hp_name not in before or hp_schema != before[hp_name]
    }
    result = {
        "properties": {
            "hyperparams": {"allOf": [{"type": "object", "properties": hp_diff}]}
        }
    }
    return result


def _top_schemas_to_hparams(top_level_schemas) -> JSON_TYPE:
    if not isinstance(top_level_schemas, dict):
        return {}
    return top_level_schemas.get("properties", {}).get("hyperparams", {})


def _hparams_schemas_to_props(hparams_schemas) -> JSON_TYPE:
    if not isinstance(hparams_schemas, dict):
        return {}
    return hparams_schemas.get("allOf", [{}])[0].get("properties", {})


def _top_schemas_to_hp_props(top_level_schemas) -> JSON_TYPE:
    hparams = _top_schemas_to_hparams(top_level_schemas)
    props = _hparams_schemas_to_props(hparams)
    return props


def _op_to_json_rec(
    op: "lale.operators.Operator",
    cls2label: Dict[str, str],
    gensym: GenSym,
    add_custom_default: bool,
) -> Tuple[str, JSON_TYPE]:
    jsn: JSON_TYPE = {}
    jsn["class"] = op.class_name()
    jsn["state"] = _get_state(op)
    if isinstance(op, lale.operators.IndividualOp):
        jsn["operator"] = op.name()
        jsn["label"] = cls2label.get(op.class_name(), op.name())
        uid = gensym(_camelCase_to_snake(jsn["label"]))
        documentation_url = op.documentation_url()
        if documentation_url is not None:
            jsn["documentation_url"] = documentation_url
        if isinstance(op, lale.operators.TrainableIndividualOp):
            if hasattr(op._impl, "viz_label"):
                jsn["viz_label"] = op._impl.viz_label()
            hyperparams = op.reduced_hyperparams()
            if hyperparams is None:
                jsn["hyperparams"] = {} if hasattr(op._impl, "fit") else None
            else:
                hp_schema = _hparams_schemas_to_props(op.hyperparam_schema())
                hyperparams = {
                    k: v
                    for k, v in hyperparams.items()
                    if not hp_schema.get(k, {}).get("transient", False)
                }
                steps: Dict[str, JSON_TYPE] = {}
                jsn["hyperparams"] = _hps_to_json_rec(
                    hyperparams, cls2label, gensym, steps, add_custom_default
                )
                if len(steps) > 0:
                    jsn["steps"] = steps
            jsn["is_frozen_trainable"] = op.is_frozen_trainable()
        if isinstance(op, lale.operators.TrainedIndividualOp):
            if hasattr(op._impl, "fit"):
                jsn["coefs"] = "coefs_not_available"
            else:
                jsn["coefs"] = None
            jsn["is_frozen_trained"] = op.is_frozen_trained()
        orig_schemas = lale.operators.get_lib_schemas(op.impl_class)
        if op._schemas is not orig_schemas:
            jsn["customize_schema"] = _get_customize_schema(op._schemas, orig_schemas)
            if add_custom_default and isinstance(
                jsn.get("customize_schema", None), dict
            ):
                if isinstance(jsn.get("hyperparams", None), dict):
                    assert jsn["hyperparams"] is not None  # to help pyright
                    orig = _top_schemas_to_hp_props(orig_schemas)
                    cust = _top_schemas_to_hp_props(jsn["customize_schema"])
                    for hp_name, hp_schema in cust.items():
                        if "default" in hp_schema:
                            if hp_name not in jsn["hyperparams"]:
                                cust_default = hp_schema["default"]
                                if hp_name in orig and "default" in orig[hp_name]:
                                    orig_default = orig[hp_name]["default"]
                                    if cust_default != orig_default:
                                        jsn["hyperparams"][hp_name] = cust_default
                                else:
                                    jsn["hyperparams"][hp_name] = cust_default
    elif isinstance(op, lale.operators.BasePipeline):
        uid = gensym("pipeline")
        child2uid: Dict[lale.operators.Operator, str] = {}
        child2jsn: Dict[lale.operators.Operator, JSON_TYPE] = {}
        for idx, child in enumerate(op.steps_list()):
            child_uid, child_jsn = _op_to_json_rec(
                child, cls2label, gensym, add_custom_default
            )
            child2uid[child] = child_uid
            child2jsn[child] = child_jsn
        jsn["edges"] = [[child2uid[x], child2uid[y]] for x, y in op.edges()]
        jsn["steps"] = {child2uid[z]: child2jsn[z] for z in op.steps_list()}
    elif isinstance(op, lale.operators.OperatorChoice):
        jsn["operator"] = "OperatorChoice"
        uid = gensym("choice")
        jsn["state"] = "planned"
        jsn["steps"] = {}
        for step in op.steps_list():
            child_uid, child_jsn = _op_to_json_rec(
                step, cls2label, gensym, add_custom_default
            )
            jsn["steps"][child_uid] = child_jsn
    else:
        raise ValueError(f"Unexpected argument of type: {type(op)}")
    return uid, jsn


def to_json(
    op: "lale.operators.Operator",
    call_depth: int = 1,
    add_custom_default: bool = False,
) -> JSON_TYPE:
    from lale.settings import disable_hyperparams_schema_validation

    cls2label = _get_cls2label(call_depth + 1)
    gensym = _init_gensym(op, cls2label)
    uid, jsn = _op_to_json_rec(op, cls2label, gensym, add_custom_default)
    if not disable_hyperparams_schema_validation:
        jsonschema.validate(jsn, SCHEMA, jsonschema.Draft4Validator)
    return jsn


def _hps_from_json_rec(jsn: Any, steps: JSON_TYPE) -> Any:
    if isinstance(jsn, dict):
        if "$ref" in jsn:
            step_uid = jsn["$ref"].split("/")[-1]
            step_jsn = steps[step_uid]
            return _op_from_json_rec(step_jsn)
        else:
            return {k: _hps_from_json_rec(v, steps) for k, v in jsn.items()}
    elif isinstance(jsn, tuple):
        return tuple([_hps_from_json_rec(v, steps) for v in jsn])
    elif isinstance(jsn, list):
        return [_hps_from_json_rec(v, steps) for v in jsn]
    else:
        return jsn


def _op_from_json_rec(jsn: JSON_TYPE) -> "lale.operators.Operator":
    kind = json_op_kind(jsn)
    if kind == "Pipeline":
        steps_dict = {uid: _op_from_json_rec(jsn["steps"][uid]) for uid in jsn["steps"]}
        steps = [steps_dict[i] for i in steps_dict]
        edges = [(steps_dict[x], steps_dict[y]) for (x, y) in jsn["edges"]]
        return lale.operators.make_pipeline_graph(steps, edges)
    elif kind == "OperatorChoice":
        steps = [_op_from_json_rec(s) for s in jsn["steps"].values()]
        name = jsn["operator"]
        return lale.operators.OperatorChoice(steps, name)
    else:
        assert kind == "IndividualOp"
        full_class_name = jsn["class"]
        last_period = full_class_name.rfind(".")
        module = importlib.import_module(full_class_name[:last_period])
        impl = getattr(module, full_class_name[last_period + 1 :])
        schemas = lale.operators.get_lib_schemas(impl)
        name = jsn["operator"]
        result = lale.operators.make_operator(impl, schemas, name)
        if jsn.get("customize_schema", {}) != {}:
            new_hps = _top_schemas_to_hp_props(jsn["customize_schema"])
            result = result.customize_schema(**new_hps)
        if jsn["state"] in ["trainable", "trained"]:
            if _get_state(result) == "planned":
                hps = jsn["hyperparams"]
                if hps is None:
                    result = result()
                else:
                    hps = _hps_from_json_rec(hps, jsn.get("steps", {}))
                    result = result(**hps)
            trnbl = cast(lale.operators.TrainableIndividualOp, result)
            if jsn["is_frozen_trainable"] and not trnbl.is_frozen_trainable():
                trnbl = trnbl.freeze_trainable()
            assert jsn["is_frozen_trainable"] == trnbl.is_frozen_trainable()
            result = trnbl
        if jsn["state"] == "trained":
            if jsn["coefs"] == "coefs_not_available":
                logger.warning(
                    f"Since the JSON representation of trained operator {name} lacks coefficients, from_json returns a trainable operator instead."
                )
            else:
                assert jsn["coefs"] is None, jsn["coefs"]
        assert (
            _get_state(result) == jsn["state"]
            or jsn["state"] == "trained"
            and jsn["coefs"] == "coefs_not_available"
        )
        assert result.documentation_url() == jsn["documentation_url"]
        return result
    assert False, f"unexpected JSON {jsn}"


def from_json(jsn: JSON_TYPE) -> "lale.operators.Operator":
    jsonschema.validate(jsn, SCHEMA, jsonschema.Draft4Validator)
    result = _op_from_json_rec(jsn)
    return result
