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

import itertools
import logging
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import jsonschema

from .schema_ranges import SchemaRange
from .schema_utils import (
    JsonSchema,
    SFalse,
    STrue,
    is_false_schema,
    is_lale_any_schema,
    is_true_schema,
    isForOptimizer,
    makeAllOf,
    makeAnyOf,
    makeOneOf,
)
from .type_checking import always_validate_schema

logger = logging.getLogger(__name__)

# Goal: given a json schema, convert it into an equivalent json-schema
# in "grouped-dnf" form:
# allOf: [anyOf: nochoice], where
# nochoice
#
# initial version, which does not try to group things intelligently:
# allOf [anyOf [P1 P2], anyOf[Q1 Q2]] ==
# anyOf [map allOf [Ps]x[Pqs]]
# Note that P1 == anyOf [P] == allOf [P]

# Given a schema, if it is an anyof, return the list of choices.
# Otherwise, return a singleton choice -- the schema

# enumerations should logically be sets.
# However, the keys are not hashable
VV = TypeVar("VV")


class set_with_str_for_keys(Generic[VV]):
    """ This mimicks a set, but uses the string representation
        of the elements for comparison tests.
        It can be used for unhashable elements, as long
        as the str function is injective
    """

    _elems: Dict[str, VV]

    def __init__(self, elems: Union[Dict[str, VV], Iterator[VV]]):
        if isinstance(elems, dict):
            self._elems = elems
        else:
            self._elems = {str(v): v for v in elems}

    def __iter__(self):
        return iter(self._elems.values())

    def __bool__(self):
        return bool(self._elems)

    def __str__(self):
        return str(list(self._elems.values()))

    def union(self, *others):
        return set_with_str_for_keys(
            [elem for subl in [self] + list(others) for elem in subl]
        )

    def intersection(self, *others):
        d: Dict[str, VV] = dict(self._elems)
        for ssk in others:
            for k in list(d.keys()):
                if k not in ssk._elems:
                    del d[k]
        return set_with_str_for_keys(d)

    def difference(self, *others):
        d: Dict[str, VV] = dict(self._elems)
        for ssk in others:
            for k in list(d.keys()):
                if k in ssk._elems:
                    del d[k]
        return set_with_str_for_keys(d)


def toAnyOfList(schema: JsonSchema) -> List[JsonSchema]:
    if "anyOf" in schema:
        return schema["anyOf"]
    else:
        return [schema]


def toAllOfList(schema: JsonSchema) -> List[JsonSchema]:
    if "allOf" in schema:
        return schema["allOf"]
    else:
        return [schema]


def liftAllOf(schemas: List[JsonSchema]) -> Iterator[JsonSchema]:
    """ Given a list of schemas, if any of them are
        allOf schemas, lift them out to the top level
    """
    for sch in schemas:
        schs2 = toAllOfList(sch)
        for s in schs2:
            yield s


def liftAnyOf(schemas: List[JsonSchema]) -> Iterator[JsonSchema]:
    """ Given a list of schemas, if any of them are
        anyOf schemas, lift them out to the top level
    """
    for sch in schemas:
        schs2 = toAnyOfList(sch)
        for s in schs2:
            yield s


# This is a great function for a breakpoint :-)
def impossible() -> JsonSchema:
    return SFalse


def enumValues(
    es: set_with_str_for_keys[Any], s: JsonSchema
) -> set_with_str_for_keys[Any]:
    """Given an enumeration set and a schema, return all the consistent values of the enumeration."""
    # TODO: actually check.  This should call the json schema validator
    ret = list()
    for e in es:
        try:
            always_validate_schema(e, s)
            ret.append(e)
        except jsonschema.ValidationError:
            logger.debug(
                f"enumValues: {e} removed from {es} because it does not validate according to {s}"
            )
    return set_with_str_for_keys(iter(ret))


# invariants for all the simplify* functions:
# - invariant: if floatAny then at most the top level return value will be 'anyOf'
# - invariant: if there is no (nested or top level) 'anyOf' then the result will not have any either

extra_field_names: List[str] = ["default", "description"]


def hasAllOperatorSchemas(schemas: List[JsonSchema]):
    if not schemas:
        return False
    for s in schemas:
        if "anyOf" in s:
            if not hasAnyOperatorSchemas(s["anyOf"]):
                return False
        elif "allOf" in s:
            if not hasAllOperatorSchemas(s["allOf"]):
                return False
        else:
            to = s.get("laleType", None)
            if to != "operator":
                return False
    return True


def hasAnyOperatorSchemas(schemas: List[JsonSchema]):
    for s in schemas:
        if "anyOf" in s:
            if hasAnyOperatorSchemas(s["anyOf"]):
                return True
        elif "allOf" in s:
            if hasAllOperatorSchemas(s["allOf"]):
                return True
        else:
            to = s.get("laleType", None)
            if to == "operator":
                return True
    return False


def simplifyAll(schemas: List[JsonSchema], floatAny: bool) -> JsonSchema:
    # First, we partition the schemas into the different types
    # that we care about
    combined_original_schema: JsonSchema = {"allOf": schemas}
    s_all: List[JsonSchema] = schemas
    s_any: List[List[JsonSchema]] = []
    s_one: List[JsonSchema] = []

    s_not: List[JsonSchema] = []
    s_not_number_list: List[
        JsonSchema
    ] = []  # a list of schemas that are a top level 'not' with a type='integer' or 'number' under it

    s_not_enum_list: List[set_with_str_for_keys[Any]] = []
    s_enum_list: List[set_with_str_for_keys[Any]] = []

    s_type: Optional[str] = None
    s_type_for_optimizer: Optional[str] = None
    s_typed: List[JsonSchema] = []
    s_other: List[JsonSchema] = []
    s_not_for_optimizer: List[JsonSchema] = []
    s_extra: Dict[str, Any] = {}

    while s_all:
        l: List[JsonSchema] = s_all
        s_all = []
        s: JsonSchema
        for s in l:
            if s is None:
                continue
            s = simplify(s, floatAny)
            if s is None:
                continue
            if not isForOptimizer(s):
                logger.info(
                    f"simplifyAll: skipping not for optimizer {s} (after simplification)"
                )
                s_not_for_optimizer.append(s)
                continue
            if is_true_schema(s):
                continue
            if is_false_schema(s):
                return SFalse
            if is_lale_any_schema(s):
                continue
            if "allOf" in s:
                s_all.extend(s["allOf"])
            elif "anyOf" in s:
                s_any.append(s["anyOf"])
            elif "oneOf" in s:
                s_one.append(s)
            elif "not" in s:
                snot = s["not"]
                if snot is None:
                    continue
                elif "enum" in snot:
                    ev = enumValues(
                        set_with_str_for_keys(snot["enum"]),
                        {"not": combined_original_schema},
                    )
                    s_not_enum_list.append(ev)
                elif "type" in snot and (
                    snot["type"] == "number" or snot["type"] == "integer"
                ):
                    s_not_number_list.append(s)
                else:
                    s_not.append(s)
            elif "enum" in s:
                ev = enumValues(
                    set_with_str_for_keys(s["enum"]), combined_original_schema
                )
                if ev:
                    s_enum_list.append(ev)
                    for k in extra_field_names:
                        if k in s:
                            d = s[k]
                            if k in s_extra and s_extra[k] != d:
                                logger.info(
                                    f"mergeAll: conflicting {k} fields: {s_extra[k]} and {d} found when merging schemas {schemas}"
                                )
                            else:
                                s_extra[k] = d
                else:
                    logger.info(
                        f"simplifyAll: {schemas} is not a satisfiable list of conjoined schemas because the enumeration {list(s['enum'])} has no elements that are satisfiable by the conjoined schemas"
                    )
                    return impossible()
            elif "type" in s:
                t = s.get("type", None)
                to = s.get("laleType", None)
                if t == "array":
                    # tuples are distinct from arrays
                    if to is not None and to == "tuple":
                        t = to

                if s_type:
                    # handle subtyping relation between integers and numbers
                    if (
                        s_type == "number"
                        and t == "integer"
                        or s_type == "integer"
                        and t == "number"
                    ):
                        s_type = "integer"
                    elif s_type != t:
                        logger.info(
                            f"simplifyAll: {schemas} is not a satisfiable list of conjoined schemas because {s} has type '{t}' and a previous schema had type '{s_type}'"
                        )
                        return impossible()
                else:
                    s_type = t
                s_typed.append(s)
            elif "XXX TODO XXX" in s and len(s) == 1:
                # Ignore missing constraints
                pass
            else:
                to = s.get("laleType", None)
                if to is None:
                    logger.warning(f"simplifyAll: '{s}' has unknown type")
                s_other.append(s)
            to = s.get("laleType", None)
            if to == "operator":
                if (
                    s_type_for_optimizer is not None
                    and s_type_for_optimizer != "operator"
                ):
                    logger.error(
                        f"simplifyAll: '{s}' has operator type for optimizer, but we also have another type for optimizer saved"
                    )
                s_type_for_optimizer = to

    # Now that we have partitioned things
    # Note: I am sure some of our assumptions here are not correct :-(, but this should do for now :-)

    # let's try to find a quick contradiction
    if s_not or s_not_number_list:
        # a bit of a special case here (which should eventually be replaced by more prinicipalled logic):
        # if one of the not cases is identical to to one of the extra cases
        # then this entire case is impossible.
        # This provides a workaround to #42 amongst other problems

        # first gather the set of extras
        pos_k: Set[str] = set()
        pk: JsonSchema
        for pk in s_typed:
            pos_k.add(str(pk))

        for sn in itertools.chain(s_not, s_not_number_list):
            snn = sn["not"]
            if str(snn) in pos_k:
                logger.info(
                    f"simplifyAll: Contradictory schema {str(combined_original_schema)} contains both {str(snn)} and its negation"
                )
                return impossible()

    # first, we simplify enumerations

    s_enum: Optional[set_with_str_for_keys[Any]] = None
    s_not_enum: Optional[set_with_str_for_keys[Any]] = None

    if s_enum_list:
        # if there are enumeration constraints, we want their intersection
        s_enum = set_with_str_for_keys.intersection(*s_enum_list)
        if not s_enum:
            # This means that enumeration values where specified
            # but none are possible, so this schema is impossible to satisfy
            logger.info(
                f"simplifyAll: {schemas} is not a satisfiable list of conjoined schemas because the conjugation of these enumerations {list(s_enum_list)} is unsatisfiable (the intersection is empty)"
            )
            return impossible()
    if s_not_enum_list:
        s_not_enum = set_with_str_for_keys.union(*s_not_enum_list)

    if s_enum and s_not_enum:
        s_enum_diff = set_with_str_for_keys.difference(s_enum, s_not_enum)
        if not s_enum_diff:
            # This means that enumeration values where specified
            # but none are possible, so this schema is impossible to satisfy
            logger.info(
                f"simplifyAll: {schemas} is not a satisfiable list of conjoined schemas because the conjugation of the enumerations is {s_enum} all of which are excluded by the conjugation of the disallowed enumerations {s_not_enum}"
            )
            return impossible()
        s_enum = s_enum_diff
        s_not_enum = None

    # break out, combine, and keep 'extra' fields, like description
    if s_typed:
        s_typed = [s.copy() for s in s_typed]
        for o in s_typed:
            for k in extra_field_names:
                if k in o:
                    d = o[k]
                    if k in s_extra and s_extra[k] != d:
                        logger.info(
                            f"mergeAll: conflicting {k} fields: {s_extra[k]} and {d} found when merging schemas {schemas}"
                        )
                    else:
                        s_extra[k] = d
                    del o[k]
        s_typed = [s for s in s_typed if s]

    if s_type == "number" or s_type == "integer":
        # First we combine all the positive number range schemas
        s_range = SchemaRange()
        s_range_for_optimizer = SchemaRange()

        for o in s_typed:
            o_range = SchemaRange.fromSchema(o)
            s_range &= o_range

            o_range_for_optimizer = SchemaRange.fromSchemaForOptimizer(o)
            s_range_for_optimizer &= o_range_for_optimizer
        # now let us look at negative number ranges
        # for now, we will not handle cases that would require splitting ranges
        # TODO: 42 is about handling more reasoning
        s_not_list = s_not_number_list
        s_not_number_list = []
        for s in s_not_list:
            snot = s["not"]
            o_range = SchemaRange.fromSchema(snot)

            success = s_range.diff(o_range)
            if success is None:
                logger.info(
                    f"simplifyAll: [range]: {s} is not a satisfiable schema, since it negates everything, falsifying the entire combined schema {combined_original_schema}"
                )
                return impossible()
            o_range_for_optimizer = SchemaRange.fromSchemaForOptimizer(snot)
            success2 = s_range_for_optimizer.diff(o_range_for_optimizer)
            if success2 is None:
                logger.info(
                    f"simplifyAll: [range]: {s} is not a satisfiable schema for the optimizer, since it negates everything, falsifying the entire combined schema {combined_original_schema}"
                )
                return impossible()

            elif success is False or success2 is False:
                s_not_number_list.append(s)

        # Now we look at negative enumarations.
        # for now, we will not handle cases that would require splitting ranges
        # TODO: 42 is about handling more reasoning
        if s_not_enum:
            s_cur_not_enum_list: set_with_str_for_keys[Any] = s_not_enum
            s_not_enum_l: List[Any] = []

            for s in s_cur_not_enum_list:
                if isinstance(s, (int, float)):
                    success = s_range.remove_point(s)
                    if success is None:
                        logger.info(
                            f'simplifyAll: [range]: {{"not": {{"enum": [{s}]}}}}  is not a satisfiable schema, since it negates everything, falsifying the entire combined schema {combined_original_schema}'
                        )
                        return impossible()
                    success2 = s_range_for_optimizer.remove_point(s)
                    if success2 is None:
                        logger.info(
                            f'simplifyAll: [range]: {{"not": {{"enum": [{s}]}}}}  is not a satisfiable schema for the optimizer, since it negates everything, falsifying the entire combined schema {combined_original_schema}'
                        )
                        return impossible()

                    elif success is False or success2 is False:
                        s_not_enum_l.append(s)
            s_not_enum = set_with_str_for_keys(iter(s_not_enum_l))

        # now let us put everything back together
        number_schema = SchemaRange.to_schema_with_optimizer(
            s_range, s_range_for_optimizer
        )

        if SchemaRange.is_empty2(s_range, s_range):
            logger.info(
                f"simplifyAll: [range]: range simplification determined that the required minimum is greater than the required maximum, so the entire thing is unsatisfiable {combined_original_schema}"
            )
            # if the actual range is empty, the entire schema is invalid
            return impossible()
        elif SchemaRange.is_empty2(s_range_for_optimizer, s_range):
            number_schema["forOptimizer"] = SFalse
            logger.info(
                f"simplifyAll: [range]: range simplification determined that the required minimum for the optimizer is greater than the required maximum, so the range is being marked as not for the optimizer: {number_schema}"
            )
        elif SchemaRange.is_empty2(s_range, s_range_for_optimizer):
            number_schema["forOptimizer"] = SFalse
            logger.info(
                f"simplifyAll: [range]: range simplification determined that the required minimum is greater than the required maximum for the optimizer, so the range is being marked as not for the optimizer: {number_schema}"
            )
        elif SchemaRange.is_empty2(s_range_for_optimizer, s_range_for_optimizer):
            logger.info(
                f"simplifyAll: [range]: range simplification determined that the required minimum for the optimizer is greater than the required maximum for the optimizer, so the range is being marked as not for the optimizer: {number_schema}"
            )
            number_schema["forOptimizer"] = SFalse

        s_typed = [number_schema]

    elif s_type == "object":
        # if this is an object type, we want to merge the properties
        s_required: Set[str] = set()
        s_props: Dict[str, List[JsonSchema]] = {}
        # TODO: generalize this to handle schema types here
        s_additionalProperties = True
        # propertyNames = []

        for o in s_typed:
            o_required = o.get("required", None)
            if o_required:
                s_required = s_required.union(o_required)

            # TODO: handle empty/absent properties case
            if "properties" in o:
                o_props = o["properties"]
            else:
                o_props = {}
            o_additionalProperties = (
                "additionalProperties" not in o or o["additionalProperties"]
            )
            # safety check:
            if not o_additionalProperties:
                for p in s_required:
                    if p not in o_props:
                        # There is a required key, but our schema
                        # does not contain that key and does not allow additional properties
                        # This schema can never be satisfied, so we can simplify this whole thing to the False schema
                        logger.info(
                            f"simplifyAll: {s_typed} is not a mergable list of schemas because {o} does not have the required key '{p}' and excludes additional properties"
                        )
                        return impossible()

            # If we do not allow additional properties
            # Remove all existing properties that are
            # not in our schema
            if not o_additionalProperties:
                for p in s_props:
                    if p not in o_props:
                        del s_props[p]
            # now go through our properties and add them
            for p, pv in o_props.items():
                if p in s_props:
                    s_props[p].append(pv)
                elif s_additionalProperties:
                    s_props[p] = [pv]
            s_additionalProperties = s_additionalProperties and o_additionalProperties
        # at this point, we have aggregated the object schemas
        # for all the properties in them
        if s_required and not s_additionalProperties:
            for k in s_required:
                if k not in s_props:
                    logger.info(
                        f"simplifyAll: {s_typed} is not a mergable list of schemas because {o} requires key '{k}', which is not in earlier schemas, and an earlier schema excluded additional properties"
                    )
                    return impossible()

        merged_props = {p: simplifyAll(s_props[p], False) for p in s_props}
        if s_required:
            for k in s_required:
                if is_false_schema(merged_props.get(k, SFalse)):
                    logger.info(
                        f"simplifyAll: required key {k} is False, so the entire conjugation of schemas {schemas} is False"
                    )
                    return impossible()

        obj: Dict[Any, Any] = {}
        obj["type"] = "object"
        if merged_props:
            obj["properties"] = merged_props
        if not s_additionalProperties:
            obj["additionalProperties"] = False
        if len(s_required) != 0:
            obj["required"] = list(s_required)
        s_typed = [obj]
    elif s_type == "array" or s_type == "tuple":
        is_tuple = s_type == "tuple"
        min_size: int = 0
        max_size: Optional[int] = None
        min_size_for_optimizer: int = 0
        max_size_for_optimizer: Optional[int] = None
        longest_item_list: int = 0
        items_schemas: List[JsonSchema] = []
        item_list_entries: List[Tuple[List[JsonSchema], Optional[JsonSchema]]] = []

        for arr in s_typed:
            arr_min_size = arr.get("minItems", 0)
            min_size = max(min_size, arr_min_size)

            arr_min_size_for_optimizer = arr.get("minItemsForOptimizer", 0)
            min_size_for_optimizer = max(
                min_size_for_optimizer, arr_min_size_for_optimizer
            )

            arr_max_size = arr.get("maxItems", None)
            if arr_max_size is not None:
                if max_size is None:
                    max_size = arr_max_size
                else:
                    max_size = min(max_size, arr_max_size)

            arr_max_size_for_optimizer = arr.get("maxItemsForOptimizer", None)
            if arr_max_size_for_optimizer is not None:
                if max_size_for_optimizer is None:
                    max_size_for_optimizer = arr_max_size_for_optimizer
                else:
                    max_size_for_optimizer = min(
                        max_size_for_optimizer, arr_max_size_for_optimizer
                    )

            arr_item = arr.get("items", None)
            if arr_item is not None:
                if isinstance(arr_item, list):
                    arr_item_len = len(arr_item)
                    longest_item_list = max(longest_item_list, arr_item_len)

                    arr_additional = arr.get("additionalItems", None)
                    item_list_entries.append((arr_item, arr_additional))
                    if arr_additional is False:
                        # If we are not allowed additional elements,
                        # that effectively sets the maximum allowed length
                        if max_size is None:
                            max_size = arr_item_len
                        else:
                            max_size = min(max_size, arr_item_len)
                else:
                    items_schemas.append(arr_item)
        # We now have accurate min/max bounds, and if there are item lists
        # we know how long the longest one is
        # additionally, we have gathered up all the item (object) schemas

        ret_arr: Dict[str, Any] = {"type": "array"}
        if is_tuple:
            ret_arr["laleType"] = "tuple"
        if min_size > 0:
            ret_arr["minItems"] = min_size

        if min_size_for_optimizer > min_size:
            ret_arr["minItemsForOptimizer"] = min_size_for_optimizer

        all_items_schema: Optional[JsonSchema] = None
        if items_schemas:
            all_items_schema = simplifyAll(items_schemas, floatAny=floatAny)

        if not item_list_entries:
            # there are no list items schemas
            assert longest_item_list == 0
            if all_items_schema:
                # deal with False schemas
                if is_false_schema(all_items_schema):
                    if min_size > 0 or min_size_for_optimizer > 0:
                        return impossible()
                    else:
                        max_size = 0
                        max_size_for_optimizer = None

                ret_arr["items"] = all_items_schema
        else:
            ret_item_list_list: List[List[JsonSchema]] = [
                [] for _ in range(longest_item_list)
            ]
            additional_schemas: List[JsonSchema] = []
            for (arr_item_list, arr_additional_schema) in item_list_entries:
                for x in range(longest_item_list):
                    ils = ret_item_list_list[x]
                    if x < len(arr_item_list):
                        ils.append(arr_item_list[x])
                    elif arr_additional_schema:
                        ils.append(arr_additional_schema)
                    if all_items_schema:
                        ils.append(all_items_schema)

                if arr_additional_schema:
                    additional_schemas.append(arr_additional_schema)

            if max_size is None or max_size > longest_item_list:
                # if it is possible to have more elements
                # we constrain them as specified

                if additional_schemas:
                    if all_items_schema is not None:
                        additional_schemas.append(all_items_schema)
                    all_items_schema = simplifyAll(
                        additional_schemas, floatAny=floatAny
                    )
                if all_items_schema is not None:
                    ret_arr["additionalItems"] = all_items_schema

            ret_item_list: List[JsonSchema] = [
                simplifyAll(x, floatAny=True) for x in ret_item_list_list
            ]
            first_false: Optional[int] = None
            for i, s in enumerate(ret_item_list):
                if is_false_schema(s):
                    first_false = i
                    break
            if first_false is not None:
                if min_size > first_false or min_size_for_optimizer > first_false:
                    return impossible()
                else:
                    if max_size is None:
                        max_size = first_false
                    else:
                        max_size = min(max_size, first_false)
                    if max_size_for_optimizer is not None:
                        if max_size_for_optimizer >= max_size:
                            max_size_for_optimizer = None
                ret_item_list = ret_item_list[0:first_false]

            ret_arr["items"] = ret_item_list

        if max_size is not None:
            ret_arr["maxItems"] = max_size
        if max_size_for_optimizer is not None:
            if max_size is None or max_size_for_optimizer < max_size:
                ret_arr["maxItemsForOptimizer"] = max_size_for_optimizer

        s_typed = [ret_arr]

    # TODO: more!
    assert not s_all
    ret_all = []
    ret_main = s_extra if s_extra else {}

    if s_type_for_optimizer is not None:
        ret_main["laleType"] = s_type_for_optimizer
    if s_enum:
        # we should simplify these as for s_not_enum
        ret_main["enum"] = list(s_enum)
        # now, we do some extra work to keep 'laleType':'operator' annotations
        if s_type_for_optimizer is None:
            from lale.operators import Operator

            if all(isinstance(x, Operator) for x in s_enum):
                # All the enumeration values are operators
                # This means it is probably an operator schema
                # which might have been missed if
                # this is being allOf'ed with an anyOfList
                if s_any and all(hasAnyOperatorSchemas(s) for s in s_any):
                    ret_main["laleType"] = "operator"
        return ret_main

    if ret_main:
        if s_typed:
            s_typed[0] = {**ret_main, **s_typed[0]}
        elif s_other:
            s_other[0] = {**ret_main, **s_other[0]}
        else:
            ret_all.append(ret_main)
    if s_typed:
        ret_all.extend(s_typed)
    if s_other:
        ret_all.extend(s_other)
    if s_not_for_optimizer:
        ret_all.extend(s_not_for_optimizer)
    if s_one:
        ret_all.extend(s_one)
    if s_not_number_list:
        ret_all.extend(s_not_number_list)
    if s_not:
        ret_all.extend(s_not)

    if s_not_enum:
        # We can't do not alongside anything else
        # TODO: we should validate the list against the
        # other parts of ret_all (this would need to move down): if any elements don't validate
        # then they already would be excluded
        # we can simplify +enum's the same way
        ret_all_agg = makeAllOf(ret_all)
        s_not_enum_simpl = enumValues(s_not_enum, ret_all_agg)
        if s_not_enum_simpl:
            sne = {"not": {"enum": list(s_not_enum)}}
            ret_all.append(sne)
        else:
            logger.debug(
                f"simplifyAll: {s_not_enum} was a negated enum that was simplified away because its elements anyway don't satisfy the additional constraints {ret_all_agg}"
            )

        s_not_enum = s_not_enum_simpl

    if not floatAny:
        ret_all.extend([simplifyAny(s, False) for s in s_any])
    ret_all_schema = makeAllOf(ret_all)
    if floatAny and s_any:
        args = list(([ret_all_schema], *tuple(s_any)))
        cp = list(itertools.product(*args))
        alls = [simplifyAll(list(s), False) for s in cp]
        ret = simplifyAny(alls, False)
        return ret
    else:
        return ret_all_schema


def simplifyAny(schema: List[JsonSchema], floatAny: bool) -> JsonSchema:
    s_any = schema

    s_enum_list: List[set_with_str_for_keys[Any]] = []
    s_not_enum_list: List[set_with_str_for_keys[Any]] = []

    s_other: List[JsonSchema] = []
    s_not_for_optimizer: List[JsonSchema] = []

    while s_any:
        schema_list = s_any
        s_any = []
        for s in schema_list:
            if s is None:
                continue
            s = simplify(s, floatAny)
            if s is None:
                continue
            if not isForOptimizer(s):
                logger.info(
                    f"simplifyAny: skipping not for optimizer {s} (after simplification)"
                )
                s_not_for_optimizer.append(s)
                continue
            if is_true_schema(s):
                return STrue
            if is_false_schema(s):
                continue
            if "anyOf" in s:
                s_any.extend(s["anyOf"])
            elif "enum" in s:
                ev = enumValues(set_with_str_for_keys(s["enum"]), s)
                if ev:
                    s_enum_list.append(ev)
            elif "not" in s:
                snot = s["not"]
                if "enum" in s["not"]:
                    ev = enumValues(set_with_str_for_keys(snot["enum"]), snot)
                    if ev:
                        s_not_enum_list.append(ev)
            else:
                s_other.append(s)

    s_enum: Optional[set_with_str_for_keys[Any]] = None
    s_not_enum: Optional[set_with_str_for_keys[Any]] = None

    if s_enum_list:
        # if there are enumeration constraints, we want their intersection
        s_enum = set_with_str_for_keys.union(*s_enum_list)

    if s_not_enum_list:
        s_not_enum = set_with_str_for_keys.intersection(*s_not_enum_list)

    if s_enum and s_not_enum:
        s_not_enum = set_with_str_for_keys.difference(s_not_enum, s_enum)
        s_enum = None

    assert not s_any
    ret: List[JsonSchema] = []

    if s_enum:
        ret.append({"enum": list(s_enum)})
    if s_not_enum:
        ret.append({"not": {"enum": list(s_not_enum)}})
    ret.extend(s_other)
    ret.extend(s_not_for_optimizer)
    return makeAnyOf(ret)


def simplifyNot(schema: JsonSchema, floatAny: bool) -> JsonSchema:
    return simplifyNot_(schema, floatAny, alreadySimplified=False)


def simplifyNot_(
    schema: JsonSchema, floatAny: bool, alreadySimplified: bool = False
) -> JsonSchema:
    """alreadySimplified=true implies that schema has already been simplified"""
    if "not" in schema:
        # if there is a not/not, we can just skip it
        ret = simplify(schema["not"], floatAny)
        return ret
    elif "anyOf" in schema:
        anys = schema["anyOf"]
        alls = [{"not": s} for s in anys]
        ret = simplifyAll(alls, floatAny)
        return ret
    elif "allOf" in schema:
        alls = schema["allOf"]
        anys = [{"not": s} for s in alls]
        ret = simplifyAny(anys, floatAny)
        return ret
    elif not alreadySimplified:
        s = simplify(schema, floatAny)
        # it is possible that the result of calling simplify
        # resulted in something that we can push 'not' down into
        # so we call ourselves, being careful to avoid an infinite loop.
        return simplifyNot_(s, floatAny, alreadySimplified=True)
    else:
        return {"not": schema}


def simplify(schema: JsonSchema, floatAny: bool) -> JsonSchema:
    """ Tries to simplify a schema into an equivalent but
        more compact/simpler one.  If floatAny if true, then
        the only anyOf in the return value will be at the top level.
        Using this option may cause a combinatorial blowup in the size
        of the schema
        """
    if is_true_schema(schema):
        return STrue
    if is_false_schema(schema):
        return SFalse
    if "enum" in schema:
        # TODO: simplify the schemas by removing anything that does not validate
        # against the rest of the schema
        return schema
    if "allOf" in schema:
        ret = simplifyAll(schema["allOf"], floatAny)
        return ret
    elif "anyOf" in schema:
        ret = simplifyAny(schema["anyOf"], floatAny)
        return ret
    elif "not" in schema:
        return simplifyNot(schema["not"], floatAny)
    elif "type" in schema and schema["type"] == "object" and "properties" in schema:
        schema2 = schema.copy()
        props = {}
        all_objs = [schema2]
        # TODO: how does this interact with required?
        # {k1:s_1, k2:anyOf:[s2s], k3:anyOf:[s3s]}
        # If floatAny is true and any properties have an anyOf in them
        # we need to float it out to the top.  We can then
        # give it to simplifyAll, which does the cross product to lift
        # them out of the list
        for k, v in schema["properties"].items():
            s = simplify(v, floatAny)
            if is_false_schema(s) and "required" in schema and s in schema["required"]:
                logger.info(
                    f"simplify: required key {k} is False, so the entire schema {schema} is False"
                )
                return impossible()

            if (not is_true_schema(s)) and floatAny and "anyOf" in s:
                all_objs.append(
                    {
                        "anyOf": [
                            {"type": "object", "properties": {k: vv}}
                            for vv in s["anyOf"]
                        ]
                    }
                )
                # If we are disallowing additionalProperties, then we can't remove this property entirely

                if not schema.get("additionalProperties", True):
                    props[k] = STrue
            else:
                props[k] = s
        schema2["properties"] = props
        if len(all_objs) == 1:
            return schema2
        else:
            # The termination argument here is somewhat subtle
            s = simplifyAll(all_objs, floatAny)
            return s
    else:
        return schema


# TODO: semantically, allOf should force an intersection
# of relevantFields, yet union seems kinder to the user/more modular (at least if additionalProperties:True)
def findRelevantFields(schema: JsonSchema) -> Optional[Set[str]]:
    """Either returns the relevant fields for the schema, or None if there was none specified"""
    if "allOf" in schema:
        fields_list: List[Optional[Set[str]]] = [
            findRelevantFields(s) for s in schema["allOf"]
        ]
        real_fields_list: List[Set[str]] = [f for f in fields_list if f is not None]
        if real_fields_list:
            return set.union(*real_fields_list)
        else:
            return None
    else:
        if "relevantToOptimizer" in schema:
            return set(schema["relevantToOptimizer"])
        else:
            return None

    # does not handle nested objects and nested relevant fields well


def narrowToGivenRelevantFields(
    schema: JsonSchema, relevantFields: Set[str]
) -> JsonSchema:
    if schema is False:
        return False
    if "anyOf" in schema:
        return {
            "anyOf": [
                narrowToGivenRelevantFields(a, relevantFields) for a in schema["anyOf"]
            ]
        }
    if "allOf" in schema:
        return {
            "allOf": [
                narrowToGivenRelevantFields(a, relevantFields) for a in schema["allOf"]
            ]
        }
    if "not" in schema:
        return {"not": narrowToGivenRelevantFields(schema["not"], relevantFields)}
    if "type" in schema and schema["type"] == "object" and "properties" in schema:
        props = schema["properties"]
        new_props = {
            k: narrowToGivenRelevantFields(v, relevantFields)
            for (k, v) in props.items()
            if k in relevantFields
        }
        schema2 = schema.copy()
        schema2["properties"] = new_props
        if "required" in schema:
            reqs = set(schema["required"])
            schema2["required"] = list(reqs.intersection(relevantFields))
        return schema2
    else:
        return schema


def narrowToRelevantFields(schema: JsonSchema) -> JsonSchema:
    relevantFields: Optional[Set[str]] = findRelevantFields(schema)
    if relevantFields is not None:
        return narrowToGivenRelevantFields(schema, relevantFields)
    else:
        return schema


# Given a json schema, removes any elements marked as 'forOptimizer:false'
# also does some basic simplifications
def filterForOptimizer(schema: JsonSchema) -> Optional[JsonSchema]:
    if schema is None or is_true_schema(schema) or is_false_schema(schema):
        return schema
    if not isForOptimizer(schema):
        return None
    if "anyOf" in schema:
        subs = schema["anyOf"]
        sch = [filterForOptimizer(s) for s in subs]
        sch_nnil = [s for s in sch if s is not None]
        if sch_nnil:
            return makeAnyOf(sch_nnil)
        else:
            return None
    if "allOf" in schema:
        subs = schema["allOf"]
        sch = [filterForOptimizer(s) for s in subs]
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
        sch = [filterForOptimizer(s) for s in subs]
        sch_nnil = [s for s in sch if s is not None]
        if sch_nnil:
            return makeOneOf(sch_nnil)
        else:
            return None

    if "not" in schema:
        s = filterForOptimizer(schema["not"])
        if s is None:
            return None
        else:
            return {"not": s}
    if "type" in schema and schema["type"] == "object" and "properties" in schema:
        # required = schema.get("required", None)

        props = {}
        for k, v in schema["properties"].items():
            s = filterForOptimizer(v)
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


def narrowSimplifyAndFilter(schema: JsonSchema, floatAny: bool) -> Optional[JsonSchema]:
    n_schema = narrowToRelevantFields(schema)
    simplified_schema = simplify(n_schema, floatAny)
    filtered_schema = filterForOptimizer(simplified_schema)
    return filtered_schema
