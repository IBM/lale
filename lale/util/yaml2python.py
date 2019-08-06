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

import ast
import astunparse
import json
import re
import sys
import yaml

def json_pprint(data, stream=sys.stdout):
    s1 = json.dumps(data)
    tree = ast.parse(s1)
    s2 = astunparse.unparse(tree)
    s3 = re.sub(r',\n\s+([\]}][^ ])', r'\1', s2)
    stream.write(s3)

data = yaml.load(sys.stdin)
json_pprint(data)
