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

from typing import Any, Iterator, List, Optional


class VisitorPathError(ValueError):

    _path: List[Any]

    def __init__(self, path: List[Any], message: Optional[str] = None):
        super().__init__(message)

        self._path = path

    def push_parent_path(self, part: Any) -> None:
        self._path.append(part)

    @property
    def path(self) -> Iterator[Any]:
        return reversed(self._path)

    def get_message_str(self) -> str:
        return super().__str__()

    def path_string(self) -> str:
        return "->".join(map(str, self.path))

    def __str__(self):
        pstr = self.path_string()
        return f"[{pstr}]: {self.get_message_str()}"
