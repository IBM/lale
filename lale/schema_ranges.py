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

from typing import Any, Optional, Union

from .schema_utils import JsonSchema


class SchemaRange(object):
    def __init__(
        self,
        minimum=None,
        maximum=None,
        exclusive_minimum=False,
        exclusive_maximum=False,
        is_integer: bool = False,
    ) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.exclusive_minimum = exclusive_minimum
        self.exclusive_maximum = exclusive_maximum
        self.is_integer = is_integer

    def __str__(self):
        res = ""
        if self.minimum is None:
            res += "(infty"
        else:
            if self.exclusive_minimum:
                res += "("
            else:
                res += "["
            res += str(self.minimum)
        res += ","
        if self.maximum is None:
            res += "infty)"
        else:
            res += str(self.maximum)
            if self.exclusive_maximum:
                res += ")"
            else:
                res += "]"
        return res

    @classmethod
    def point(cls, pt: Union[int, float]):
        return SchemaRange(
            minimum=pt,
            maximum=pt,
            exclusive_minimum=False,
            exclusive_maximum=False,
            is_integer=isinstance(pt, int),
        )

    @classmethod
    def fromSchema(cls, schema: Any) -> "SchemaRange":
        return SchemaRange(
            minimum=schema.get("minimum", None),
            maximum=schema.get("maximum", None),
            exclusive_minimum=schema.get("exclusiveMinimum", False),
            exclusive_maximum=schema.get("exclusiveMaximum", False),
            is_integer=schema.get("type", "number") == "integer",
        )

    @classmethod
    def fromSchemaForOptimizer(cls, schema: Any) -> "SchemaRange":
        s = cls.fromSchema(schema)

        minimum = schema.get("minimumForOptimizer", None)
        maximum = schema.get("maximumForOptimizer", None)
        exclusive_minimum = schema.get("exclusiveMinimumForOptimizer", False)
        exclusive_maximum = schema.get("exclusiveMaximumForOptimizer", False)
        is_integer = (
            schema.get("type", "numberForOptimizer") == "integer" or s.is_integer
        )

        if minimum is None:
            minimum = s.minimum
            if s.minimum is not None:
                exclusive_minimum = exclusive_minimum or s.exclusive_minimum
        elif s.minimum is not None and minimum == s.minimum:
            exclusive_minimum = exclusive_minimum or s.exclusive_minimum

        if maximum is None:
            maximum = s.maximum
            if s.maximum is not None:
                exclusive_maximum = exclusive_maximum or s.exclusive_maximum
        elif s.maximum is not None and minimum == s.minimum:
            exclusive_maximum = exclusive_maximum or s.exclusive_maximum

        return SchemaRange(
            minimum=minimum,
            maximum=maximum,
            exclusive_minimum=exclusive_minimum,
            exclusive_maximum=exclusive_maximum,
            is_integer=is_integer,
        )

    @classmethod
    def to_schema_with_optimizer(
        cls, actual_range: "SchemaRange", optimizer_range: "SchemaRange"
    ) -> JsonSchema:
        number_schema: JsonSchema = {}
        if actual_range.is_integer:
            number_schema["type"] = "integer"
        else:
            number_schema["type"] = "number"
            if optimizer_range.is_integer:
                number_schema["laleType"] = "integer"

        if actual_range.minimum is not None:
            number_schema["minimum"] = actual_range.minimum
            if actual_range.exclusive_minimum:
                number_schema["exclusiveMinimum"] = True

        if optimizer_range.minimum is not None:
            if (
                actual_range.minimum is None
                or actual_range.minimum < optimizer_range.minimum
                or (
                    actual_range.minimum == optimizer_range.minimum
                    and optimizer_range.exclusive_minimum
                    and not actual_range.exclusive_minimum
                )
            ):
                number_schema["minimumForOptimizer"] = optimizer_range.minimum
                if optimizer_range.exclusive_minimum:
                    number_schema["exclusiveMinimumForOptimizer"] = True

        if actual_range.maximum is not None:
            number_schema["maximum"] = actual_range.maximum
            if actual_range.exclusive_maximum:
                number_schema["exclusiveMaximum"] = True

        if optimizer_range.maximum is not None:
            if (
                actual_range.maximum is None
                or actual_range.maximum > optimizer_range.maximum
                or (
                    actual_range.maximum == optimizer_range.maximum
                    and optimizer_range.exclusive_maximum
                    and not actual_range.exclusive_maximum
                )
            ):
                number_schema["maximumForOptimizer"] = optimizer_range.maximum
                if optimizer_range.exclusive_maximum:
                    number_schema["exclusiveMaximumForOptimizer"] = True

        return number_schema

    def __iand__(self, other: "SchemaRange"):
        self.is_integer = self.is_integer or other.is_integer
        if other.minimum is not None:
            if self.minimum is None:
                self.minimum = other.minimum
                self.exclusive_minimum = other.exclusive_minimum
            elif self.minimum == other.minimum:
                self.exclusive_minimum = (
                    self.exclusive_minimum or other.exclusive_minimum
                )
            elif self.minimum < other.minimum:
                self.minimum = other.minimum
                self.exclusive_minimum = other.exclusive_minimum

        if other.maximum is not None:
            if self.maximum is None:
                self.maximum = other.maximum
                self.exclusive_maximum = other.exclusive_maximum
            elif self.maximum == other.maximum:
                self.exclusive_maximum = (
                    self.exclusive_maximum or other.exclusive_maximum
                )
            elif self.maximum > other.maximum:
                self.maximum = other.maximum
                self.exclusive_maximum = other.exclusive_maximum

        return self

    def diff(self, other: "SchemaRange") -> Optional[bool]:
        """Returns None if the resulting region is impossible.
        Returns True if the other constraint was completely subtracted from
        self.  If it could not be, then it returns False (and the caller should probably
        keep the other constraint as a negated constraint)
        """
        # for now, just handle simple exclusions
        if not other.is_integer or other.is_integer == self.is_integer:
            # the exclusion is less than the actual range
            if (
                self.minimum is not None
                and other.maximum is not None
                and (
                    other.maximum < self.minimum
                    or (
                        other.maximum == self.minimum
                        and (self.exclusive_minimum or other.exclusive_maximum)
                    )
                )
            ):
                return True
            # the exclusion is greater than the actual range
            if (
                self.maximum is not None
                and other.minimum is not None
                and (
                    other.minimum > self.maximum
                    or (
                        other.minimum == self.maximum
                        and (self.exclusive_maximum or other.exclusive_minimum)
                    )
                )
            ):
                return True

            if other.minimum is None:
                if self.minimum is None:
                    # the exclusion and the range have no minimum
                    if other.maximum is None:
                        # nothing is possible
                        return None
                    else:
                        self.minimum = other.maximum
                        self.exclusive_minimum = not other.exclusive_maximum
                        return True
                # else might create a hole, so avoid this case
            else:
                # ASSERT: other.minimum is not None
                if (
                    self.minimum is None
                    or self.minimum < other.minimum
                    or (
                        self.minimum == other.minimum
                        and (not self.exclusive_minimum or other.exclusive_minimum)
                    )
                ):
                    if (
                        other.maximum is None
                        or self.maximum is not None
                        and (
                            other.maximum > self.maximum
                            or (
                                (
                                    other.maximum == self.maximum
                                    and (
                                        not other.exclusive_maximum
                                        or self.exclusive_maximum
                                    )
                                )
                            )
                        )
                    ):
                        self.maximum = other.minimum
                        self.exclusive_maximum = not other.exclusive_minimum
                        return True
                    # else might create a hole, so avoid this case
                else:
                    # self.minimum >= other.minimum
                    if (
                        other.maximum is None
                        or self.maximum < other.maximum
                        or (
                            self.maximum == other.maximum
                            and (not other.exclusive_maximum or self.exclusive_maximum)
                        )
                    ):
                        # nothing is possible
                        return None
                    else:
                        self.minimum = other.maximum
                        self.exclusive_minimum = not other.exclusive_maximum
                        return True

            if other.maximum is None:
                if self.maximum is None:
                    # the exclusion and the range have no maximum
                    if other.minimum is None:
                        # nothing is possible
                        return None
                    else:
                        self.maximum = other.minimum
                        self.exclusive_maximum = not other.exclusive_minimum
                        return True
                # else might create a hole, so avoid this case
            else:
                # ASSERT: other.maximum is not None
                if (
                    self.maximum is None
                    or self.maximum > other.maximum
                    or (
                        self.maximum == other.maximum
                        and (not self.exclusive_maximum or other.exclusive_maximum)
                    )
                ):
                    if (
                        other.minimum is None
                        or self.minimum is not None
                        and (
                            other.minimum < self.minimum
                            or (
                                (
                                    other.minimum == self.minimum
                                    and (
                                        not other.exclusive_minimum
                                        or self.exclusive_minimum
                                    )
                                )
                            )
                        )
                    ):
                        self.minimum = other.maximum
                        self.exclusive_minimum = not other.exclusive_maximum
                        return True
                    # else might create a hole, so avoid this case
                else:
                    # self.maximum >= other.maximum
                    if (
                        other.minimum is None
                        or self.minimum > other.minimum
                        or (
                            self.minimum == other.minimum
                            and (not other.exclusive_minimum or self.exclusive_minimum)
                        )
                    ):
                        # nothing is possible
                        return None
                    else:
                        self.maximum = other.minimum
                        self.exclusive_maximum = not other.exclusive_minimum
                        return True
        return False

    def remove_point(self, other: Union[int, float]) -> Optional[bool]:
        """Returns None if the resulting region is impossible.
        Returns True if the other constraint was completely subtracted from
        self.  If it could not be, then it returns False (and the caller should probably
        keep the other constraint as a negated constraint)
        """
        return self.diff(SchemaRange.point(other))

    @classmethod
    def is_empty2(cls, lower: "SchemaRange", upper: "SchemaRange") -> bool:
        """Determines if the range given by taking lower bounds from lower and upper bound from upper is empty (contains nothing)
        is_integer is assumed to be their disjunction
        """
        is_integer = lower.is_integer or upper.is_integer

        if lower.minimum is not None and upper.maximum is not None:
            if lower.minimum > upper.maximum:
                return True
            if lower.minimum == upper.maximum and (
                lower.exclusive_minimum or upper.exclusive_maximum
            ):
                return True
            if (
                is_integer
                and lower.exclusive_minimum
                and upper.exclusive_maximum
                and lower.minimum + 1 == upper.maximum
            ):
                return True
        return False

    def is_empty(self) -> bool:
        """Determines if the range is empty (contains nothing)"""
        return SchemaRange.is_empty2(self, self)
