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

import subprocess
import tempfile
import unittest
import os


class TestNotebooks(unittest.TestCase):
   pass


def create_test(path):
    def exec_notebook(self):
        with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
            args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                    "--ExecutePreprocessor.timeout=1000",
                    "--output", fout.name, path]
            subprocess.check_call(args)
    return exec_notebook

for filename in os.listdir('examples'):
    if filename.lower().endswith('.ipynb'):
        test_name = 'test_notebook_{0}'.format(filename[:-len('.ipynb')])
        test_method = create_test('examples/'+filename)
        setattr(TestNotebooks, test_name, test_method)
