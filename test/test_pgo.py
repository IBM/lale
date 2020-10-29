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

import unittest
import warnings
from typing import Sequence

import lale.lib.lale
import lale.search.PGO as PGO
from lale.lib.sklearn import PCA, LogisticRegression
from lale.search.lale_grid_search_cv import get_grid_search_parameter_grids
from lale.search.op2hp import hyperopt_search_space

example_pgo_fp = "test/lale-pgo-example.json"


class TestPGOLoad(unittest.TestCase):
    def test_pgo_load(self):
        pgo = PGO.load_pgo_file(example_pgo_fp)
        _ = pgo["LogisticRegression"]["C"]

    def test_pgo_sample(self):
        pgo = PGO.load_pgo_file(example_pgo_fp)
        lr_c = pgo["LogisticRegression"]["C"]
        dist = PGO.FrequencyDistribution.asIntegerValues(lr_c.items())
        samples: Sequence[int] = dist.samples(10)
        _ = samples


#        print(f"LR[C] samples: {samples}")


class TestPGOGridSearchCV(unittest.TestCase):
    def test_lr_parameters(self):
        pgo = PGO.load_pgo_file(example_pgo_fp)

        lr = LogisticRegression()
        parameters = get_grid_search_parameter_grids(lr, num_samples=2, pgo=pgo)
        _ = parameters

    #        print(parameters)

    def test_lr_run(self):
        pgo = PGO.load_pgo_file(example_pgo_fp)

        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer

        lr = LogisticRegression()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = lale.lib.lale.GridSearchCV(
                estimator=lr,
                lale_num_samples=2,
                lale_num_grids=5,
                cv=5,
                pgo=pgo,
                scoring=make_scorer(accuracy_score),
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)

    def test_pipeline_parameters(self):
        pgo = PGO.load_pgo_file(example_pgo_fp)

        trainable = PCA() >> LogisticRegression()
        parameters = get_grid_search_parameter_grids(trainable, num_samples=2, pgo=pgo)
        _ = parameters
        # print(parameters)


class TestPGOHyperopt(unittest.TestCase):
    def test_lr_parameters(self):
        pgo = PGO.load_pgo_file(example_pgo_fp)

        lr = LogisticRegression()
        hp_search_space = hyperopt_search_space(lr, pgo=pgo)
        _ = hp_search_space

    def test_lr_run(self):
        pgo = PGO.load_pgo_file(example_pgo_fp)

        from sklearn.datasets import load_iris

        from lale.lib.lale import Hyperopt

        lr = LogisticRegression()
        clf = Hyperopt(estimator=lr, max_evals=5, pgo=pgo)
        iris = load_iris()
        clf.fit(iris.data, iris.target)

    def test_pipeline_parameters(self):
        pgo = PGO.load_pgo_file(example_pgo_fp)

        trainable = PCA() >> LogisticRegression()
        parameter_grids = get_grid_search_parameter_grids(
            trainable, num_samples=2, pgo=pgo
        )
        _ = parameter_grids

        # print(parameters)
