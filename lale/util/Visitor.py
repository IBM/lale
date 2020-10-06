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

from typing import Any


class Visitor(object):
    def defaultVisit(self, node, *args, **kwargs):
        raise NotImplementedError

    def __getattr__(self, attr):
        if attr.startswith("visit"):
            return self.defaultVisit
        return self.__getattribute__(attr)

    def _visitAll(self, iterable, *args, **kwargs):
        def filter(x):
            return (x is not None) or None

        return [filter(x) and accept(x, self, *args, **kwargs) for x in iterable]


# Because of the magic way we add accept methods, mypy does not know they exist
# so this method is important for accept calls to typecheck
def accept(obj: Any, v: Visitor, *args, **kwargs):
    return obj._accept(v, *args, **kwargs)
