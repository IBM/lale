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

import enum
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .schema_utils import JsonSchema, SchemaEnum

logger = logging.getLogger(__name__)


class DiscoveredEnums(object):
    def __init__(
        self,
        enums: Optional[SchemaEnum] = None,
        children: Optional[Dict[str, "DiscoveredEnums"]] = None,
    ) -> None:
        self.enums = enums
        self.children = children

    def __str__(self) -> str:
        def val_as_str(v):
            if v is None:
                return "null"
            elif isinstance(v, str):
                return f"'{v}'"
            else:
                return str(v)

        en = ""
        if self.enums:
            ens = [val_as_str(v) for v in self.enums]
            en = ", ".join(sorted(ens))

        ch = ""
        if self.children:
            chs = [f"{str(k)}->{str(v)}" for k, v in self.children.items()]
            ch = ",".join(chs)

        if en and ch:
            en = en + "; "

        return "<" + en + ch + ">"


def schemaToDiscoveredEnums(schema: JsonSchema) -> Optional[DiscoveredEnums]:
    """ Given a schema, returns a positive enumeration set.
    This is very conservative, and even includes negated enum constants
    (since the assumption is that they may, in some contexts, be valid)
    """

    def combineDiscoveredEnums(
        combine: Callable[[Iterable[SchemaEnum]], Optional[SchemaEnum]],
        des: Iterable[Optional[DiscoveredEnums]],
    ) -> Optional[DiscoveredEnums]:
        enums: List[SchemaEnum] = []
        children: Dict[str, List[DiscoveredEnums]] = {}
        for de in des:
            if de is None:
                continue
            if de.enums is not None:
                enums.append(de.enums)
            if de.children is not None:
                for cn, cv in de.children.items():
                    if cv is None:
                        continue
                    if cn in children:
                        children[cn].append(cv)
                    else:
                        children[cn] = [cv]

        combined_enums: Optional[SchemaEnum] = None
        if enums:
            combined_enums = combine(enums)

        if not children:
            if combined_enums is None:
                return None
            else:
                return DiscoveredEnums(enums=combined_enums)
        else:
            combined_children: Dict[str, DiscoveredEnums] = {}
            for ccn, ccv in children.items():
                if not ccv:
                    continue

                ccvc = combineDiscoveredEnums(combine, ccv)
                if ccvc is not None:
                    combined_children[ccn] = ccvc
            return DiscoveredEnums(enums=combined_enums, children=combined_children)

    def joinDiscoveredEnums(
        des: Iterable[Optional[DiscoveredEnums]],
    ) -> Optional[DiscoveredEnums]:
        def op(args: Iterable[SchemaEnum]) -> Optional[SchemaEnum]:
            return set.union(*args)

        return combineDiscoveredEnums(op, des)

    def meetDiscoveredEnums(
        des: Tuple[Optional[DiscoveredEnums], ...]
    ) -> Optional[DiscoveredEnums]:
        def op(args: Iterable[SchemaEnum]) -> Optional[SchemaEnum]:
            return set.intersection(*args)

        return combineDiscoveredEnums(op, des)

    if schema is True or schema is False:
        return None
    if "enum" in schema:
        # TODO: we should validate the enum elements according to the schema, like schema2search_space does
        return DiscoveredEnums(enums=set(schema["enum"]))
    if "type" in schema:
        typ = schema["type"]
        if typ == "object" and "properties" in schema:
            props = schema["properties"]
            pret: Dict[str, DiscoveredEnums] = {}
            for p, s in props.items():
                pos = schemaToDiscoveredEnums(s)
                if pos is not None:
                    pret[p] = pos
            if pret:
                return DiscoveredEnums(children=pret)
            else:
                return None
        else:
            return None

    if "not" in schema:
        neg = schemaToDiscoveredEnums(schema["not"])
        return neg

    if "allOf" in schema:
        posl = [schemaToDiscoveredEnums(s) for s in schema["allOf"]]
        return joinDiscoveredEnums(posl)

    if "anyOf" in schema:
        posl = [schemaToDiscoveredEnums(s) for s in schema["anyOf"]]
        return joinDiscoveredEnums(posl)

    if "oneOf" in schema:
        posl = [schemaToDiscoveredEnums(s) for s in schema["oneOf"]]
        return joinDiscoveredEnums(posl)
    return None


def accumulateDiscoveredEnumsToPythonEnums(
    de: Optional[DiscoveredEnums], path: List[str], acc: Dict[str, enum.Enum]
) -> None:
    def withEnumValue(e: str) -> Tuple[str, Any]:
        if isinstance(e, str):
            return (e.replace("-", "_"), e)
        elif isinstance(e, (int, float, complex)):
            return ("num" + str(e), e)
        else:
            logger.info(
                f"Unknown type ({type(e)}) of enumeration constant {e}, not handling very well"
            )
            return (str(e), e)

    if de is None:
        return
    if de.enums is not None:
        ppath, _ = withEnumValue("_".join(path))
        epath = ".".join(path)
        acc[ppath] = enum.Enum(
            epath, (withEnumValue(x) for x in de.enums if x is not None)
        )
    if de.children is not None:
        for k in de.children:
            accumulateDiscoveredEnumsToPythonEnums(de.children[k], [k] + path, acc)


def discoveredEnumsToPythonEnums(de: Optional[DiscoveredEnums]) -> Dict[str, enum.Enum]:
    acc: Dict[str, enum.Enum] = {}
    accumulateDiscoveredEnumsToPythonEnums(de, [], acc)
    return acc


def schemaToPythonEnums(schema: JsonSchema) -> Dict[str, enum.Enum]:
    de = schemaToDiscoveredEnums(schema)
    enums = discoveredEnumsToPythonEnums(de)
    return enums


def addDictAsFields(obj: Any, d: Dict[str, Any], force=False) -> None:
    if d is None:
        return
    for k, v in d.items():
        if k == "":
            logger.warning(
                f"There was a top level enumeration specified, so it is not being added to {obj._name}"
            )
        elif hasattr(obj, k) and not force:
            logger.error(
                f"The object {obj._name} already has the field {k}.  This conflicts with our attempt at adding that key as an enumeration field"
            )
        else:
            setattr(obj, k, v)


def addSchemaEnumsAsFields(obj: Any, schema: JsonSchema, force=False) -> None:
    enums = schemaToPythonEnums(schema)
    addDictAsFields(obj, enums, force)
