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

import yaml
from lale import helpers
from lale.search import schema2search_space as opt
from lale.search.HP import search_space_to_hp_expr, search_space_to_hp_str
from lale.search.op2hp import hyperopt_search_space

from .. import schema2enums as enums
from lale.lib.sklearn import LogisticRegression

def evaluate_and_print(name, dirname, filename):
    schema = helpers.load_yaml(dirname, filename, meta_dir=False)
    relevantFields = opt.findRelevantFields(schema)
    if relevantFields:
        schema = opt.narrowToGivenRelevantFields(schema, relevantFields)
    simplified_schema = opt.simplify(schema, True)
#    helpers.print_yaml('SIMPLIFIED_' + name, simplified_schema)
    filtered_schema = opt.filterForOptimizer(simplified_schema)
#    helpers.print_yaml('FILTERED_' + name, filtered_schema)

    hp_s = opt.schemaToSearchSpaceHelper(name, filtered_schema, relevantFields)
    if hp_s:
        e = search_space_to_hp_expr(hp_s, name)
#        grids = optCV.hpToGSGrids(hp_s)
#        print(str(grids))

    print("hyperopt:")
    print(search_space_to_hp_str(hp_s, name))

tests_dir = r'lale/tests/schemas'
concat_dir = r'meta_data/ConcatFeatures'
nystroem_dir = r'meta_data/Nystroem'
one_hot_encoder_dir = r'meta_data/OneHotEncoder'
pca_dir = r'meta_data/PCA'

hyperparameter_filename = r'hyperparameter_schema.yaml'

def test1():
    filename = r'test1.yaml'
    evaluate_and_print("test1", tests_dir, filename)

def test4():
    filename = r'test4.yaml'
    evaluate_and_print("test4", tests_dir, filename)

def test_array1():
    filename = r'test_array1.yaml'
    evaluate_and_print("test_array1", tests_dir, filename)

def test_forOptimizer_concat_schema():
    filename = r'forOptimizer_concat.yaml'
    evaluate_and_print("forOptimizer_concat", tests_dir, filename)

def test_concat_hyperparameter_schema():
    evaluate_and_print("concat", concat_dir, hyperparameter_filename)

def test_nystroem_hyperparameter_schema():
    evaluate_and_print("nystroem", nystroem_dir, hyperparameter_filename)

def test_one_hot_encoder_hyperparameter_schema():
    evaluate_and_print("one_hot_encoder", one_hot_encoder_dir, hyperparameter_filename)

def test_pca_hyperparameter_schema():
    evaluate_and_print("pca", pca_dir, hyperparameter_filename)

def test_with_hyperopt():
    from test.test_with_hyperopt import HyperoptClassifierTest
    clf = HyperoptClassifierTest(hyperopt_search_space(LogisticRegression))
    clf.test_classifier()
    print(str(list(LogisticRegression.class_weight)))

def test_with_hyperopt_params():
    from test.test_with_hyperopt import HyperoptClassifierTest
    lr = LogisticRegression(solver=LogisticRegression.solver.liblinear)
    clf = HyperoptClassifierTest(hyperopt_search_space(lr))
    clf.test_classifier()
    print(str(list(LogisticRegression.class_weight)))
    print(str(LogisticRegression.penalty.l1))

def test_concat_with_hyperopt():
    from test.test_with_hyperopt import HyperoptMetaModelTest
    from lale.lib.lale import ConcatFeatures
    from lale.lib.sklearn import LogisticRegression
    from lale.lib.sklearn import Nystroem
    from lale.lib.sklearn import PCA

    pca = PCA(n_components=3)
    nys = Nystroem(n_components=10)
    concat = ConcatFeatures()
    lr = LogisticRegression(random_state=42, C=0.1)

    trainable = (pca & nys) >> concat >> lr
    clf = HyperoptMetaModelTest(trainable)
    clf.test_meta_model()

def test_instances_with_hyperopt():
    from test.test_with_hyperopt import HyperoptMetaModelTest

    from lale.lib.sklearn import LogisticRegression
    from lale.lib.sklearn import PCA
    tfm = PCA(n_components=3)
    clf = LogisticRegression(random_state=42)
    trainable = tfm >> clf

    clf = HyperoptMetaModelTest(trainable)
    clf.test_meta_model()

def test_mlp_with_hyperopt():
    from lale.lib.sklearn import MLPClassifier
    from test.test_with_hyperopt import HyperoptClassifierTest
    clf = HyperoptClassifierTest(hyperopt_search_space(MLPClassifier))
    clf.test_classifier()
