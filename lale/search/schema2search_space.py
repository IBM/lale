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

import math
import logging
import numpy
import jsonschema


from typing import Any, Dict, List, Set, Iterable, Iterator, Optional, Tuple, Union
from lale.schema_simplifier import findRelevantFields, narrowToGivenRelevantFields, simplify, filterForOptimizer

from lale.schema_utils import Schema, getMinimum, getMaximum, STrue, SFalse, is_false_schema, is_true_schema, forOptimizer
from lale.search.search_space import *
from lale.search.HP import search_space_to_str_for_comparison
from lale.search.PGO import PGO, FrequencyDistribution, Freqs

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def get_default(schema)->Optional[Any]:
    d = schema.get('default', None)
    if d is not None:
        try:
            s = forOptimizer(schema)
            jsonschema.validate(d, s)
            return d
        except:
            logger.debug(f"get_default: default {d} not used because it is not valid for the schema {schema}")
            return None
    return None

class FreqsWrapper(object):
    base:Optional[Dict[str,Freqs]]

    def __init__(self, base:Optional[Dict[str,Freqs]]):
        self.base = base

def pgo_lookup(pgo:Optional[PGO], name:str)->Optional[FreqsWrapper]:
    if pgo is None:
        return None
    else:
        freqs:Optional[Dict[str,Freqs]] = None
        if pgo is not None:
            freqs = pgo.get(name, None)
        return FreqsWrapper(freqs)

pgo_part = Union[FreqsWrapper, Freqs, None]

def freqs_wrapper_lookup(part:pgo_part, k:str)->pgo_part:
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

def asFreqs(part:pgo_part)->Optional[Iterable[Tuple[Any, int]]]:
    if part is None:
        return None
    elif isinstance(part, FreqsWrapper):
        return None
    else:
        return part.items()

def schemaObjToSearchSpaceHelper(
    longName:str, 
    path:str, 
    schema:Schema, 
    relevantFields:Optional[Set[str]],
    pgo_freqs:pgo_part=None)->Dict[str,SearchSpace]:
        if 'properties' not in schema:
            return {}
        props = schema['properties']
        hyp:Dict[str, SearchSpace] = {}
        for p,s in props.items():
            if relevantFields is None or p in relevantFields:
                # TODO: This does not handle nested relevant fields correctly
                # We would need to specify what is correct in that case
                sub_freqs = freqs_wrapper_lookup(pgo_freqs, p)
                sub_sch = schemaToSearchSpaceHelper_(longName, path + "_" + p, s, relevantFields, pgo_freqs=sub_freqs)
                if sub_sch is None:
                    # if it is a required field, this entire thing should be None
                    hyp[p] = SearchSpaceConstant(None)
                else:
                    hyp[p] = sub_sch
            else:
                logger.debug(f"schemaToSearchSpace: skipping not relevant field {p}")
        return hyp

def schemaToSearchSpaceHelper_( longName, 
                                path:str, 
                                schema:Schema, 
                                relevantFields:Optional[Set[str]],
                                pgo_freqs:pgo_part=None)->Optional[SearchSpace]:
    # TODO: handle degenerate cases
    # right now, this handles only a very fixed form

    if is_false_schema(schema):
        return None

    if 'enum' in schema:
        vals = schema['enum']
        return SearchSpaceEnum(vals, pgo=asFreqs(pgo_freqs), default=get_default(schema))

    if 'type' in schema:
        typ = schema['type']
        if typ == "boolean":
            return SearchSpaceBool(pgo=asFreqs(pgo_freqs), default=get_default(schema))
        elif typ == "number" or typ == "integer":
            exclusive_minimum = False
            minimum=schema.get('minimumForOptimizer', None)
            if minimum is not None:
                exclusive_minimum = schema.get('exclusiveMinimumForOptimizer', False)
            else:
                minimum=schema.get('minimum', None)
                if minimum is not None:
                    exclusive_minimum = schema.get('exclusiveMinimum', False)

            exclusive_maximum = False
            maximum=schema.get('maximumForOptimizer', None)
            if maximum is not None:
                exclusive_maximum = schema.get('exclusiveMaximumForOptimizer', False)
            else:
                maximum=schema.get('maximum', None)
                if maximum is not None:
                    exclusive_maximum = schema.get('exclusiveMaximum', False)

            distribution = schema.get('distribution', None)

            typeForOptimizer = schema.get('typeForOptimizer', None)
            if typeForOptimizer is None:
                typeForOptimizer = typ

            if typeForOptimizer == "number":
                discrete = False
            elif typeForOptimizer == "integer":
                discrete = True
            else:
                raise NotImplementedError()
            
            pgo:Freqs

            return SearchSpaceNumber(minimum=minimum, 
                            exclusiveMinimum=exclusive_minimum, 
                            maximum=maximum, 
                            exclusiveMaximum=exclusive_maximum, 
                            discrete=discrete, 
                            distribution=distribution,
                            pgo=asFreqs(pgo_freqs),
                            default=get_default(schema))
        elif typ == "array" or typ =="tuple":
            typeForOptimizer = schema.get('typeForOptimizer', None)
            if typeForOptimizer is None:
                typeForOptimizer = typ

            is_tuple:bool = typeForOptimizer == "tuple"

            items_schema = schema.get('itemsForOptimizer', None)
            if items_schema is None:
                items_schema = schema.get('items', None)
                if items_schema is None:
                    raise ValueError(f"an array type was found without a provided schema for the items in the schema {schema}.  Please provide a schema for the items (consider using itemsForOptimizer)")

            if isinstance(items_schema, list):
                contents = []
                for i,sub_schema in enumerate(items_schema):
                    sub = schemaToSearchSpaceHelper_(longName, path + "_" + str(i), sub_schema, relevantFields)
                    if sub is None:
                        return None
                    else:
                        contents.append(sub)
                return SearchSpaceList(contents=contents, is_tuple=is_tuple)

            min_items = schema.get('minItemsForOptimizer', None)
            if min_items is None:
                min_items = schema.get('minItems', None)
                if min_items is None:
                    min_items = 0
            max_items = schema.get('maxItemsForOptimizer', None)
            if max_items is None:
                max_items = schema.get('maxItems', None)
                if max_items is None:
                    raise ValueError(f"an array type was found without a provided maximum number of items in the schema {schema}.  Please provide a maximum (consider using maxItemsForOptimizer)")

            sub_opt = schemaToSearchSpaceHelper_(longName, path + "-", items_schema, relevantFields)
            is_tuple = typeForOptimizer == "tuple"
            if sub_opt is None:
                if min_items <= 0 and max_items > 0:
                    return SearchSpaceConstant([])
                else:
                    return None
            else:
                return SearchSpaceArray(minimum=min_items, maximum=max_items, contents=sub_opt, is_tuple=is_tuple)

        elif typ == "object":
            if 'properties' not in schema:
                return SearchSpaceObject(longName, [], [])
            o = schemaObjToSearchSpaceHelper(longName, path, schema, relevantFields, pgo_freqs=pgo_freqs)
            all_keys = list(o.keys())
            all_keys.sort()
            o_choice = tuple([o.get(k, None) for k in all_keys])
            return SearchSpaceObject(longName, all_keys, [o_choice])
        elif typ == "string":
            pass
        else:
            raise ValueError(f"An unknown type ({typ}) was found in the schema {schema}")


    if 'anyOf' in schema:
        objs = []
        for s_obj in schema['anyOf']:
            if 'type' in s_obj and s_obj['type'] == "object":
                o = schemaObjToSearchSpaceHelper(longName, path, s_obj, relevantFields, pgo_freqs=pgo_freqs)
                if o: objs.append(o)
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
            anys:Dict[str,Any] = {}
            for o in objs:
                o_choice = tuple([o.get(k, None) for k in all_keys])
                k = str([as_str(all_keys[idx], c) for idx, c in enumerate(o_choice)])
                if k in anys:
                    logger.info(f"Ignoring Duplicate SearchSpace entry {k}")
                anys[k] = o_choice
            return SearchSpaceObject(longName, all_keys, anys.values())
        else:
            return SearchSpaceObject(longName, [], [])
    
    if 'allOf' in schema:
        # if all but one are negated constraints, we will just ignore them
        pos_sub_schema:List[Schema] = []
        for sub_schema in schema['allOf']:
            if 'not' not in sub_schema:
                pos_sub_schema.append(sub_schema)

        if len(pos_sub_schema) > 1:
            raise ValueError(f"schemaToSearchSpaceHelper does not yet know how to compile the given schema {schema} for {longName}, because it is an allOf with more than one non-negated schemas ({pos_sub_schema})")
        if len(pos_sub_schema) == 0:
            raise ValueError(f"schemaToSearchSpaceHelper does not yet know how to compile the given schema {schema} for {longName}, because it is an allOf with only negated schemas")
        
        logger.debug(f"schemaToSearchSpaceHelper: ignoring negated schemas in the conjunction {schema} for {longName}")
        return schemaToSearchSpaceHelper_(longName, 
                                path,
                                pos_sub_schema[0], 
                                relevantFields,
                                pgo_freqs=pgo_freqs)
    # TODO: handle degenerate cases
    raise ValueError(f"schemaToSearchSpaceHelper does not yet know how to compile the given schema {schema} for {longName}")

def schemaToSearchSpaceHelper(longName, 
                              schema:Schema, 
                              relevantFields:Optional[Set[str]],
                              pgo_freqs:pgo_part=None)->Optional[SearchSpace]:
    if not is_false_schema(schema) and not schema:
        return None
    else:
        return schemaToSearchSpaceHelper_(longName, longName, schema, relevantFields, pgo_freqs=pgo_freqs)

def schemaToSimplifiedAndSearchSpace(
    longName:str, 
    name:str, 
    schema:Schema,
    pgo:Optional[PGO]=None)->Tuple[Schema, Optional[SearchSpace]]:
    relevantFields = findRelevantFields(schema)
    if relevantFields:
        schema = narrowToGivenRelevantFields(schema, relevantFields)
    simplified_schema = simplify(schema, True)
#    from . import helpers
#    helpers.print_yaml('SIMPLIFIED_' + longName, simplified_schema)

    filtered_schema = filterForOptimizer(simplified_schema)
#    helpers.print_yaml('FILTERED_' + longName, filtered_schema)

    return (filtered_schema, 
            schemaToSearchSpaceHelper(
                        longName, 
                        filtered_schema, 
                        relevantFields, 
                        pgo_freqs=pgo_lookup(pgo, name)))

def schemaToSearchSpace(longName:str, 
                        name:str, 
                        schema:Schema, 
                        pgo:Optional[PGO]=None)->Optional[SearchSpace]:
    (s, h) = schemaToSimplifiedAndSearchSpace(longName, name, schema, pgo=pgo)
    return h
