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

import os
import subprocess
import tempfile
import warnings
from typing import List, Optional

import pytest


def should_test(f: str) -> bool:
    notebooks_categories_str: Optional[str] = os.environ.get("NOTEBOOK_CATEGORY", None)
    notebooks_categories: Optional[
        List[str]
    ] = notebooks_categories_str.split() if notebooks_categories_str is not None else None
    all_notebooks_categories_str: Optional[str] = os.environ.get(
        "ALL_NOTEBOOK_CATEGORIES", None
    )
    all_notebooks_categories: Optional[
        List[str]
    ] = all_notebooks_categories_str.split() if all_notebooks_categories_str is not None else None

    if notebooks_categories is None:
        if all_notebooks_categories is None:
            # run everything (with a warning)
            warnings.warn(
                "Running all notebook tests.  To run a subset, specify appropriate filters using the NOTEBOOK_CATEGORY and ALL_NOTEBOOK_CATEGORIES environment variables"
            )
            return True
        else:
            # we want to run all tests that are *not* in the all list
            # this is for running the stuff left over (in another job) after we carve out prefixes with NOTEBOOK_CATEGORY
            for c in all_notebooks_categories:
                if f.startswith(c):
                    return False
            return True
    else:
        if all_notebooks_categories is not None:
            # check that the category is included in the master list, if set (useful for travis)
            # if the list of categories is not set, continue (useful for running on the command line)
            for c in notebooks_categories:
                assert c in all_notebooks_categories
        for c in notebooks_categories:
            if f.startswith(c):
                return True
        # if it is not a requested category, don't run it
        return False


@pytest.mark.parametrize(
    "filename",
    [f for f in os.listdir("examples") if f.endswith(".ipynb") and should_test(f)],
)
def test_notebook(filename):
    path = os.path.join("examples", filename)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=1000",
            "--output",
            fout.name,
            path,
        ]
        subprocess.check_call(args)
