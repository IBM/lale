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

from typing import Any, Dict, List, Optional, Set, Union

# Type definitions
JsonSchema = Dict[str, Any]
SchemaEnum = Set[Any]

STrue: JsonSchema = {}
SFalse: JsonSchema = {"not": STrue}

forOptimizerConstant: str = "forOptimizer"
forOptimizerConstantSuffix: str = "ForOptimizer"


def is_true_schema(s: JsonSchema) -> bool:
    return s is True or s == STrue


def is_false_schema(s: JsonSchema) -> bool:
    return s is False or s == SFalse


def is_lale_any_schema(s: JsonSchema) -> bool:
    if isinstance(s, dict):
        t = s.get("laleType", None)
        return t == "Any"
    else:
        return False


def getForOptimizer(obj, prop: str):
    return obj.get(prop + forOptimizerConstantSuffix, None)


def getMinimum(obj):
    prop = "minimum"
    m = obj.get(prop)
    mfo = getForOptimizer(obj, prop)
    if mfo is None:
        return m
    else:
        if m is not None and mfo < m:
            raise ValueError(
                f"A minimum ({m}) and a *smaller* minimumForOptimizer ({mfo}) was specified in {obj}"
            )
        return mfo


def getMaximum(obj):
    prop = "maximum"
    m = obj.get(prop)
    mfo = getForOptimizer(obj, prop)
    if mfo is None:
        return m
    else:
        if m is not None and mfo > m:
            raise ValueError(
                f"A maximum ({m}) and a *greater* maximumForOptimizer ({mfo}) was specified in {obj}"
            )
        return mfo


def getExclusiveMinimum(obj):
    prop = "exclusveMinimum"
    m = obj.get(prop)
    mfo = getForOptimizer(obj, prop)
    if mfo is None:
        return m
    else:
        return mfo


def getExclusiveMaximum(obj):
    prop = "exclusiveMaximum"
    m = obj.get(prop)
    mfo = getForOptimizer(obj, prop)
    if mfo is None:
        return m
    else:
        return mfo


def isForOptimizer(s: JsonSchema) -> bool:
    if isinstance(s, dict):
        return s.get(forOptimizerConstant, True)
    else:
        return True


def makeSingleton_(k: str, schemas: List[JsonSchema]) -> JsonSchema:
    if len(schemas) == 0:
        return {}
    if len(schemas) == 1:
        return schemas[0]
    else:
        return {k: schemas}


def makeAllOf(schemas: List[JsonSchema]) -> JsonSchema:
    return makeSingleton_("allOf", schemas)


def makeAnyOf(schemas: List[JsonSchema]) -> JsonSchema:
    return makeSingleton_("anyOf", schemas)


def makeOneOf(schemas: List[JsonSchema]) -> JsonSchema:
    return makeSingleton_("oneOf", schemas)


def forOptimizer(schema: JsonSchema) -> Optional[JsonSchema]:
    if schema is None or schema is True or schema is False:
        return schema
    if not isForOptimizer(schema):
        return None
    if "anyOf" in schema:
        subs = schema["anyOf"]
        sch = [forOptimizer(s) for s in subs]
        sch_nnil = [s for s in sch if s is not None]
        if sch_nnil:
            return makeAnyOf(sch_nnil)
        else:
            return None
    if "allOf" in schema:
        subs = schema["allOf"]
        sch = [forOptimizer(s) for s in subs]
        sch_nnil = [s for s in sch if s is not None]
        filtered_sch = sch_nnil
        if len(sch_nnil) != len(sch):
            # Questionable semantics here (aka HACK!!!!)
            # Since we removed something from the schema
            # we will also remove negated schemas
            filtered_sch = [
                s for s in sch_nnil if not isinstance(s, dict) or "not" not in s
            ]

        if filtered_sch:
            return makeAllOf(filtered_sch)
        else:
            return None
    if "oneOf" in schema:
        subs = schema["oneOf"]
        sch = [forOptimizer(s) for s in subs]
        sch_nnil = [s for s in sch if s is not None]
        if sch_nnil:
            return makeOneOf(sch_nnil)
        else:
            return None

    if "not" in schema:
        s = forOptimizer(schema["not"])
        if s is None:
            return None
        else:
            return {"not": s}

    transformedSchema: JsonSchema = {}
    for k, v in schema.items():
        if k.endswith(forOptimizerConstantSuffix):
            base: str = k[: -len(forOptimizerConstantSuffix)]
            transformedSchema[base] = v
        elif k not in transformedSchema:
            transformedSchema[k] = v

    schema = transformedSchema
    if "type" in schema and schema["type"] == "object" and "properties" in schema:
        # required = schema.get("required", None)

        props = {}
        for k, v in schema["properties"].items():
            s = forOptimizer(v)
            if s is None:
                #                if required and k in required:
                # if this field is required (and has now been filtered)
                # filter the whole object schema
                return None
            else:
                props[k] = s

        ret = schema.copy()
        ret["properties"] = props
        return ret

    return schema


def has_operator(schema: JsonSchema) -> bool:
    to = schema.get("laleType", None)
    if to == "operator":
        return True
    if "not" in schema:
        if has_operator(schema["not"]):
            return True
    if "anyOf" in schema:
        if any(has_operator(s) for s in schema["anyOf"]):
            return True
    if "allOf" in schema:
        if any(has_operator(s) for s in schema["allOf"]):
            return True
    if "oneOf" in schema:
        if any(has_operator(s) for s in schema["oneOf"]):
            return True
    if "items" in schema:
        it = schema["items"]
        if isinstance(it, list):
            if any(has_operator(s) for s in it):
                return True
        else:
            if has_operator(it):
                return True
    if "properties" in schema:
        props = schema["properties"]
        if any(has_operator(s) for s in props.values()):
            return True
    if "patternProperties" in schema:
        pattern_props = schema["patternProperties"]
        if any(has_operator(s) for s in pattern_props.values()):
            return True
    if "additionalProperties" in schema:
        add_props = schema["additionalProperties"]
        if not isinstance(add_props, bool):
            if has_operator(add_props):
                return True
    if "dependencies" in schema:
        depends = schema["dependencies"]
        for d in depends.values():
            if not isinstance(d, list):
                if has_operator(d):
                    return True
    # if we survived all of the checks, then we
    return False


def atomize_schema_enumerations(
    schema: Union[None, JsonSchema, List[JsonSchema]]
) -> None:
    """ Given a schema, converts structured enumeration values (records, arrays)
        into schemas where the structured part is specified as a schema, with the
        primitive as the enum.
    """
    if schema is None:
        return
    if isinstance(schema, list):
        for s in schema:
            atomize_schema_enumerations(s)
        return

    if not isinstance(schema, dict):
        return

    for key in ["anyOf", "allOf", "oneOf", "items", "additionalProperties", "not"]:
        atomize_schema_enumerations(schema.get(key, None))

    for key in ["properties", "patternProperties", "dependencies"]:
        v = schema.get(key, None)
        if v is not None:
            atomize_schema_enumerations(list(v.values()))

    # now that we have handled all the recursive cases
    ev = schema.get("enum", None)
    if ev is not None:
        simple_evs: List[Any] = []
        complex_evs: List[JsonSchema] = []
        for e in ev:
            if isinstance(e, dict):
                required: List[str] = []
                props: Dict[str, JsonSchema] = {}
                for k, v in e.items():
                    required.append(k)
                    vs = {"enum": [v]}
                    atomize_schema_enumerations(vs)
                    props[k] = vs
                ds = {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(e.keys()),
                    "properties": props,
                }
                complex_evs.append(ds)

            elif isinstance(e, list) or isinstance(e, tuple):
                is_tuple = isinstance(e, tuple)
                items_len = len(e)
                items: List[JsonSchema] = []
                for v in e:
                    vs = {"enum": [v]}
                    atomize_schema_enumerations(vs)
                    items.append(vs)

                ls = {
                    "type": "array",
                    "items": items,
                    "additionalItems": False,
                    "minItems": items_len,
                    "maxItems": items_len,
                }
                if is_tuple:
                    ls["laleType"] = "tuple"
                complex_evs.append(ls)
            else:
                simple_evs.append(ev)

        if complex_evs:
            del schema["enum"]
            if simple_evs:
                complex_evs.append({"enum": simple_evs})
            if len(complex_evs) == 1:
                # special case, just update in place
                schema.update(complex_evs[0])
            else:
                schema["anyOf"] = complex_evs


def check_operators_schema(
    schema: Optional[Union[List[JsonSchema], JsonSchema]], warnings: List[str]
) -> None:
    """ Given a schema, collect warnings if there
    are any enumeration with all Operator values
    that are not marked as `'laleType':'operator'`.
    This should be called after simplification.
    """
    if schema is None:
        return

    if isinstance(schema, list):
        for s in schema:
            check_operators_schema(s, warnings)
        return

    if not isinstance(schema, dict):
        return

    for key in ["anyOf", "allOf", "oneOf", "items", "additionalProperties", "not"]:
        v = schema.get(key, None)
        if v is not None:
            check_operators_schema(v, warnings)

    for key in ["properties", "patternProperties", "dependencies"]:
        v = schema.get(key, None)
        if v is not None:
            check_operators_schema(list(v.values()), warnings)

    if "enum" in schema:
        es = schema["enum"]
        if es:
            to = schema.get("laleType", None)
            if to != "operator":
                from lale.operators import Operator

                if all(isinstance(e, Operator) for e in es):
                    warnings.append(
                        f"The enumeration {[e.name() for e in es]} is all lale operators, but the schema fragment {schema} it is part of does not stipulate that it should be 'laleType':'operator'.  While legal, this likely indicates either an omission in the schema or a bug in the schema simplifier"
                    )
