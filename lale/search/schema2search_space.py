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

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import jsonschema

from lale.operators import (
    Operator,
    OperatorChoice,
    PlannedIndividualOp,
    PlannedOperator,
    PlannedPipeline,
)
from lale.schema_simplifier import (
    filterForOptimizer,
    findRelevantFields,
    narrowToGivenRelevantFields,
    simplify,
)
from lale.schema_utils import (
    JsonSchema,
    atomize_schema_enumerations,
    check_operators_schema,
    forOptimizer,
    has_operator,
    is_false_schema,
)
from lale.search.lale_hyperopt import search_space_to_str_for_comparison
from lale.search.PGO import PGO, Freqs
from lale.search.search_space import (
    SearchSpace,
    SearchSpaceArray,
    SearchSpaceBool,
    SearchSpaceConstant,
    SearchSpaceDict,
    SearchSpaceEmpty,
    SearchSpaceEnum,
    SearchSpaceNumber,
    SearchSpaceObject,
    SearchSpaceOperator,
    SearchSpaceProduct,
    SearchSpaceSum,
    should_print_search_space,
)
from lale.util import VisitorPathError
from lale.util.Visitor import Visitor, accept

logger = logging.getLogger(__name__)


class OperatorSchemaError(VisitorPathError):
    def __init__(self, sub_path: Any, message: Optional[str] = None):
        super().__init__([], message)

        self.sub_path = sub_path

    def get_message_str(self) -> str:
        msg = super().get_message_str()
        if self.sub_path is None:
            return msg
        else:
            return f"for path {self.sub_path}: {msg}"


def op_to_search_space(
    op: PlannedOperator, pgo: Optional[PGO] = None, data_schema={}
) -> SearchSpace:
    """ Given an operator, this method compiles its schemas into a SearchSpace
    """
    search_space = SearchSpaceOperatorVisitor.run(op, pgo=pgo, data_schema=data_schema)

    if should_print_search_space("true", "all", "search_space"):
        name = op.name()
        if not name:
            name = "an operator"
        print(f"search space for {name}:\n {str(search_space)}")
    return search_space


def get_default(schema) -> Optional[Any]:
    d = schema.get("default", None)
    if d is not None:
        try:
            s = forOptimizer(schema)
            jsonschema.validate(d, s, jsonschema.Draft4Validator)
            return d
        except jsonschema.ValidationError:
            logger.debug(
                f"get_default: default {d} not used because it is not valid for the schema {schema}"
            )
            return None
    return None


class FreqsWrapper(object):
    base: Optional[Dict[str, Freqs]]

    def __init__(self, base: Optional[Dict[str, Freqs]]):
        self.base = base


def pgo_lookup(pgo: Optional[PGO], name: str) -> Optional[FreqsWrapper]:
    if pgo is None:
        return None
    else:
        freqs: Optional[Dict[str, Freqs]] = None
        if pgo is not None:
            freqs = pgo.get(name, None)
        return FreqsWrapper(freqs)


pgo_part = Union[FreqsWrapper, Freqs, None]


def freqs_wrapper_lookup(part: pgo_part, k: str) -> pgo_part:
    if part is None:
        return None
    elif isinstance(part, FreqsWrapper):
        f = part.base
        if f is not None and k in f:
            return f[k]
        else:
            return None
    else:
        return None


def asFreqs(part: pgo_part) -> Optional[Iterable[Tuple[Any, int]]]:
    if part is None:
        return None
    elif isinstance(part, FreqsWrapper):
        return None
    else:
        return part.items()


def add_sub_space(space, k, v):
    """ Given a search space and a "key",
        if the defined subschema does not exist,
        set it to be the constant v space
   """
    # TODO!
    # I should parse __ and such and walk down the schema
    if isinstance(space, SearchSpaceObject):
        if k not in space.keys:
            space.keys.append(k)
            space.choices = (c + (SearchSpaceConstant(v),) for c in space.choices)
            return


# TODO: do we use 'path' above anymore?
# or do we just add the paths later as needed?
class SearchSpaceOperatorVisitor(Visitor):
    pgo: Optional[PGO]

    @classmethod
    def run(
        cls, op: PlannedOperator, pgo: Optional[PGO] = None, data_schema={}
    ) -> SearchSpace:
        visitor = cls(pgo=pgo, data_schema=data_schema)
        return accept(op, visitor)

    def __init__(self, pgo: Optional[PGO] = None, data_schema={}):
        super(SearchSpaceOperatorVisitor, self).__init__()
        self.pgo = pgo
        self.data_schema = data_schema

    def visitPlannedIndividualOp(self, op: PlannedIndividualOp) -> SearchSpace:
        schema = op._hyperparam_schema_with_hyperparams(self.data_schema)
        module = op._impl.__module__
        if module is None or module == str.__class__.__module__:
            long_name = op.name()
        else:
            long_name = module + "." + op.name()
        name = op.name()
        space = self.schemaToSearchSpace(long_name, name, schema)
        if space is None:
            space = SearchSpaceEmpty()
        # we now augment the search space as needed with the specified hyper-parameters
        # even if they are marked as not relevant to the optimizer, we still want to include them now
        if hasattr(op, "_hyperparams"):
            hyperparams = op._hyperparams
            if hyperparams:
                for (k, v) in hyperparams.items():
                    add_sub_space(space, k, v)
        return space

    visitTrainableIndividualOp = visitPlannedIndividualOp
    visitTrainedIndividualOp = visitPlannedIndividualOp

    def visitPlannedPipeline(self, op: "PlannedPipeline") -> SearchSpace:
        spaces: List[Tuple[str, SearchSpace]] = [
            (s.name(), accept(s, self)) for s in op.steps()
        ]
        return SearchSpaceProduct(spaces)

    visitTrainablePipeline = visitPlannedPipeline
    visitTrainedPipeline = visitPlannedPipeline

    def visitOperatorChoice(self, op: "OperatorChoice") -> SearchSpace:
        spaces: List[SearchSpace] = [accept(s, self) for s in op.steps()]

        return SearchSpaceSum(spaces)

    # functions to actually convert an individual operator
    # schema into a search space
    def JsonSchemaToSearchSpaceHelper(
        self,
        longName: str,
        path: str,
        schema: JsonSchema,
        relevantFields: Optional[Set[str]],
        pgo_freqs: pgo_part = None,
        sub_space: bool = True,
    ) -> Dict[str, SearchSpace]:
        if "properties" not in schema:
            return {}
        props = schema["properties"]
        hyp: Dict[str, SearchSpace] = {}
        for p, s in props.items():
            if relevantFields is None or p in relevantFields:
                # We would need to specify what is correct in that case
                sub_freqs = freqs_wrapper_lookup(pgo_freqs, p)
                sub_sch = self.schemaToSearchSpaceHelper_(
                    longName, path + "_" + p, s, None, pgo_freqs=sub_freqs
                )
                if sub_sch is None:
                    # if it is a required field, this entire thing should be None
                    hyp[p] = SearchSpaceConstant(None)
                else:
                    hyp[p] = sub_sch
            else:
                logger.debug(f"schemaToSearchSpace: skipping not relevant field {p}")
        return hyp

    def schemaToSearchSpaceHelper_(
        self,
        longName,
        path: str,
        schema: JsonSchema,
        relevantFields: Optional[Set[str]],
        pgo_freqs: pgo_part = None,
        sub_space: bool = True,
    ) -> Optional[SearchSpace]:
        # TODO: handle degenerate cases
        # right now, this handles only a very fixed form

        if is_false_schema(schema):
            return None

        typ: Optional[str] = None
        typ = schema.get("laleType", None)
        if typ is None:
            typ = schema.get("type", None)
        else:
            typ = typ

        if "enum" in schema and typ != "operator":
            vals = schema["enum"]
            return SearchSpaceEnum(
                vals, pgo=asFreqs(pgo_freqs), default=get_default(schema)
            )

        if typ is not None:
            if typ == "boolean":
                return SearchSpaceBool(
                    pgo=asFreqs(pgo_freqs), default=get_default(schema)
                )
            elif typ == "number" or typ == "integer":
                exclusive_minimum = False
                minimum = schema.get("minimumForOptimizer", None)
                if minimum is not None:
                    exclusive_minimum = schema.get(
                        "exclusiveMinimumForOptimizer", False
                    )
                else:
                    minimum = schema.get("minimum", None)
                    if minimum is not None:
                        exclusive_minimum = schema.get("exclusiveMinimum", False)

                exclusive_maximum = False
                maximum = schema.get("maximumForOptimizer", None)
                if maximum is not None:
                    exclusive_maximum = schema.get(
                        "exclusiveMaximumForOptimizer", False
                    )
                else:
                    maximum = schema.get("maximum", None)
                    if maximum is not None:
                        exclusive_maximum = schema.get("exclusiveMaximum", False)

                distribution = schema.get("distribution", None)

                laleType = schema.get("laleType", None)
                if laleType is None:
                    laleType = typ

                if laleType == "number":
                    discrete = False
                elif laleType == "integer":
                    discrete = True
                else:
                    raise OperatorSchemaError(
                        path,
                        f"specified laleType should be a number or integer, not: {laleType}.",
                    )

                pgo: Freqs

                return SearchSpaceNumber(
                    minimum=minimum,
                    exclusiveMinimum=exclusive_minimum,
                    maximum=maximum,
                    exclusiveMaximum=exclusive_maximum,
                    discrete=discrete,
                    distribution=distribution,
                    pgo=asFreqs(pgo_freqs),
                    default=get_default(schema),
                )
            elif typ == "array" or typ == "tuple":
                laleType = schema.get("laleType", None)
                if laleType is None:
                    laleType = typ

                is_tuple: bool = laleType == "tuple"

                min_items = schema.get("minItemsForOptimizer", None)
                if min_items is None:
                    min_items = schema.get("minItems", None)
                    if min_items is None:
                        min_items = 0
                max_items = schema.get("maxItemsForOptimizer", None)
                if max_items is None:
                    max_items = schema.get("maxItems", None)

                items_schema = schema.get("itemsForOptimizer", None)
                if items_schema is None:
                    items_schema = schema.get("items", None)
                    if items_schema is None:
                        raise OperatorSchemaError(
                            path,
                            f"An array type was found without a provided schema for the items in the schema {schema}. Please provide a schema for the items (consider using itemsForOptimizer)",
                        )

                # we can search an empty list even without schemas
                if max_items == 0:
                    if is_tuple:
                        return SearchSpaceConstant([()])
                    else:
                        return SearchSpaceConstant([[]])

                prefix: Optional[List[SearchSpace]] = None
                additional: Optional[SearchSpace] = None
                if isinstance(items_schema, list):
                    prefix = []
                    for i, sub_schema in enumerate(items_schema):
                        sub = self.schemaToSearchSpaceHelper_(
                            longName, path + "_" + str(i), sub_schema, relevantFields
                        )
                        if sub is None:
                            return None
                        else:
                            prefix.append(sub)
                    prefix_len = len(prefix)
                    additional_items_schema = schema.get(
                        "additionalItemsForOptimizer", None
                    )
                    if additional_items_schema is None:
                        additional_items_schema = schema.get("additionalItems", None)
                    if additional_items_schema is None:
                        if max_items is None or max_items > prefix_len:
                            raise OperatorSchemaError(
                                path,
                                f"An array type was found with provided schemas for {prefix_len} elements, but either an unspecified or too high a maxItems, and no schema for the additionalItems.  Please constraing maxItems to <= {prefix_len} (you can set maxItemsForOptimizer), or provide a schema for additionalItems",
                            )
                    elif additional_items_schema is False:
                        if max_items is None:
                            max_items = prefix_len
                        else:
                            max_items = min(max_items, prefix_len)
                    else:
                        additional = self.schemaToSearchSpaceHelper_(
                            longName, path + "-", sub_schema, relevantFields
                        )
                        # if items_schema is None:
                        #     raise ValueError(f"an array type was found without a provided schema for the items in the schema {schema}.  Please provide a schema for the items (consider using itemsForOptimizer)")
                else:
                    additional = self.schemaToSearchSpaceHelper_(
                        longName, path + "-", items_schema, relevantFields
                    )

                if max_items is None:
                    raise OperatorSchemaError(
                        path,
                        f"An array type was found without a provided maximum number of items in the schema {schema}, and it is not a list with 'additionalItems' set to False.  Please provide a maximum (consider using maxItemsForOptimizer), or, if you are using a list, set additionalItems to False",
                    )

                return SearchSpaceArray(
                    prefix=prefix,
                    minimum=min_items,
                    maximum=max_items,
                    additional=additional,
                    is_tuple=is_tuple,
                )

            elif typ == "object":
                if "properties" not in schema:
                    return SearchSpaceObject(longName, [], [])
                o = self.JsonSchemaToSearchSpaceHelper(
                    longName,
                    path,
                    schema,
                    relevantFields,
                    pgo_freqs=pgo_freqs,
                    sub_space=sub_space,
                )
                if sub_space:
                    return SearchSpaceDict(o)
                else:
                    all_keys = list(o.keys())
                    all_keys.sort()
                    o_choice = tuple([o.get(k, None) for k in all_keys])
                    return SearchSpaceObject(longName, all_keys, [o_choice])

            elif typ == "string":
                pass
            elif typ == "operator":
                # TODO: If there is a default, we could use it
                vals = schema.get("enum", None)
                if vals is None:
                    logger.error(
                        "An operator is required by the schema but was not provided"
                    )
                    return None

                sub_schemas = [
                    accept(op, self)
                    if isinstance(op, Operator)
                    else SearchSpaceConstant(op)
                    for op in vals
                ]
                combined_sub_schema: SearchSpace
                if len(sub_schemas) == 1:
                    combined_sub_schema = sub_schemas[0]
                    if isinstance(combined_sub_schema, SearchSpaceConstant):
                        return combined_sub_schema
                else:
                    combined_sub_schema = SearchSpaceSum(sub_schemas)
                    if all((isinstance(x, SearchSpaceConstant) for x in sub_schemas)):
                        return combined_sub_schema
                return SearchSpaceOperator(combined_sub_schema)

            elif typ == "Any":
                raise OperatorSchemaError(
                    path,
                    f"A search space was found with laleType ({typ}), which is not searchable.  Please mark the relevant hyperparameter as not relevant for the optimizer.  schema: {schema}",
                )
            else:
                raise OperatorSchemaError(
                    path, f"An unknown type ({typ}) was found in the schema {schema}"
                )

        if "anyOf" in schema:
            objs = []
            for s_obj in schema["anyOf"]:
                if "type" in s_obj and s_obj["type"] == "object":
                    o = self.JsonSchemaToSearchSpaceHelper(
                        longName,
                        path,
                        s_obj,
                        relevantFields,
                        pgo_freqs=pgo_freqs,
                        sub_space=sub_space,
                    )
                    if o:
                        objs.append(o)
            if objs:
                # First, gather a list of all the properties
                keys_list = [set(o.keys()) for o in objs]
                # make sure the iterator is deterministic
                all_keys = list(set.union(*keys_list))
                # and we might as well make it sorted
                all_keys.sort()

                def as_str(k, c):
                    if c is None:
                        return "None"
                    else:
                        return search_space_to_str_for_comparison(c, path + "_" + k)

                anys: Dict[str, Any] = {}
                for o in objs:
                    o_choice = tuple([o.get(k, None) for k in all_keys])
                    k = str(
                        [as_str(all_keys[idx], c) for idx, c in enumerate(o_choice)]
                    )
                    if k in anys:
                        logger.info(f"Ignoring Duplicate SearchSpace entry {k}")
                    anys[k] = o_choice
                return SearchSpaceObject(longName, all_keys, anys.values())
            else:
                return SearchSpaceObject(longName, [], [])

        if "allOf" in schema:
            # if all but one are negated constraints, we will just ignore them
            pos_sub_schema: List[JsonSchema] = []
            for sub_schema in schema["allOf"]:
                if "not" not in sub_schema:
                    pos_sub_schema.append(sub_schema)

            if len(pos_sub_schema) > 1:
                raise OperatorSchemaError(
                    path,
                    f"schemaToSearchSpaceHelper does not yet know how to compile the given schema {schema}, because it is an allOf with more than one non-negated schemas ({pos_sub_schema})",
                )
            if len(pos_sub_schema) == 0:
                raise OperatorSchemaError(
                    path,
                    f"schemaToSearchSpaceHelper does not yet know how to compile the given schema {schema}, because it is an allOf with only negated schemas",
                )

            logger.debug(
                f"[{path}]: schemaToSearchSpaceHelper: ignoring negated schemas in the conjunction {schema}"
            )
            return self.schemaToSearchSpaceHelper_(
                longName,
                path,
                pos_sub_schema[0],
                relevantFields,
                pgo_freqs=pgo_freqs,
                sub_space=sub_space,
            )
        # TODO: handle degenerate cases
        raise OperatorSchemaError(
            path,
            f"schemaToSearchSpaceHelper does not yet know how to compile the given schema {schema}",
        )

    def schemaToSearchSpaceHelper(
        self,
        longName,
        schema: Optional[JsonSchema],
        relevantFields: Optional[Set[str]],
        pgo_freqs: pgo_part = None,
        sub_space: bool = True,
    ) -> Optional[SearchSpace]:
        if schema is None or is_false_schema(schema):
            return None
        else:
            if sub_space:
                relevantFields = None
            return self.schemaToSearchSpaceHelper_(
                longName,
                longName,
                schema,
                relevantFields,
                pgo_freqs=pgo_freqs,
                sub_space=sub_space,
            )

    def schemaToSimplifiedAndSearchSpace(
        self, longName: str, name: str, schema: JsonSchema
    ) -> Tuple[Optional[JsonSchema], Optional[SearchSpace]]:
        relevantFields = findRelevantFields(schema)
        if relevantFields:
            schema = narrowToGivenRelevantFields(schema, relevantFields)

        if has_operator(schema):
            atomize_schema_enumerations(schema)
        simplified_schema = simplify(schema, True)
        # from .. import pretty_print

        # print(f"SIMPLIFIED_{longName}: {pretty_print.to_string(simplified_schema)}")

        filtered_schema: Optional[JsonSchema] = filterForOptimizer(simplified_schema)
        #    print(f'SIMPLIFIED_{longName}: {pretty_print.to_string(filtered_schema)}')

        if logger.isEnabledFor(logging.WARNING):
            op_warnings: List[str] = []
            check_operators_schema(filtered_schema, op_warnings)
            if op_warnings:
                for w in op_warnings:
                    logger.warning(w)

        return (
            filtered_schema,
            self.schemaToSearchSpaceHelper(
                longName,
                filtered_schema,
                relevantFields,
                pgo_freqs=pgo_lookup(self.pgo, name),
                sub_space=False,
            ),
        )

    def schemaToSearchSpace(
        self, longName: str, name: str, schema: JsonSchema
    ) -> Optional[SearchSpace]:
        (s, h) = self.schemaToSimplifiedAndSearchSpace(longName, name, schema)
        return h
