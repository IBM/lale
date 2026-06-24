# Copyright 2019-2023 IBM Corporation
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

import os
import re
from datetime import datetime
from pathlib import Path

from setuptools import setup


def get_version() -> str:
    init_path = Path(__file__).parent / "lale" / "__init__.py"
    content = init_path.read_text()
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE)
    assert match is not None
    return match.group(1)


version_base = get_version()
version = (
    f"{version_base}-{datetime.now().strftime('%y%m%d%H%M')}"
    if "TRAVIS" in os.environ
    else version_base
)

if os.environ.get("READTHEDOCS") == "True":
    setup(version=version, install_requires=[])
else:
    setup(version=version)
