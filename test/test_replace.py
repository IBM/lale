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

from lale.lib.lale import NoOp
from lale.lib.sklearn import (
    PCA,
    RFE,
    AdaBoostRegressor,
    DecisionTreeClassifier,
    LinearRegression,
    LogisticRegression,
    SelectKBest,
    SimpleImputer,
    StackingClassifier,
    VotingClassifier,
)


class TestReplace(unittest.TestCase):
    def test_choice(self):
        two_choice = PCA | SelectKBest
        replaced_choice = two_choice.replace(PCA, LogisticRegression)
        expected_choice = LogisticRegression | SelectKBest
        self.assertEqual(replaced_choice.to_json(), expected_choice.to_json())

        three_choice = PCA | SelectKBest | NoOp
        replaced_choice = three_choice.replace(PCA, LogisticRegression)
        expected_choice = LogisticRegression | SelectKBest | NoOp
        self.assertEqual(replaced_choice.to_json(), expected_choice.to_json())

    def test_simple_pipeline(self):
        pipeline_simple = PCA >> SelectKBest >> LogisticRegression
        simple_imputer = SimpleImputer
        replaced_pipeline = pipeline_simple.replace(PCA, simple_imputer)
        expected_pipeline = SimpleImputer >> SelectKBest >> LogisticRegression
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

    def test_choice_pipeline(self):
        pipeline_choice = (PCA | NoOp) >> SelectKBest >> LogisticRegression
        simple_imputer = SimpleImputer
        replaced_pipeline = pipeline_choice.replace(PCA, simple_imputer)
        expected_pipeline = (SimpleImputer | NoOp) >> SelectKBest >> LogisticRegression
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

    def test_planned_trained_ops(self):
        pca1 = PCA(n_components=10)
        pca2 = PCA(n_components=5)
        choice = pca1 | pca2
        pipeline_choice = (pca1 | pca2) >> LogisticRegression

        replaced_choice = choice.replace(pca1, SimpleImputer)  # SimpleImputer | pca2
        expected_choice = SimpleImputer | pca2
        self.assertEqual(replaced_choice.to_json(), expected_choice.to_json())

        replaced_pipeline = pipeline_choice.replace(
            pca1, SimpleImputer
        )  # SimpleImputer | pca2
        expected_pipeline = (SimpleImputer | pca2) >> LogisticRegression
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

        replaced_choice = choice.replace(pca2, SimpleImputer)  # pca1 | SimpleImputer
        expected_choice = pca1 | SimpleImputer
        self.assertEqual(replaced_choice.to_json(), expected_choice.to_json())

        replaced_pipeline = pipeline_choice.replace(
            pca2, SimpleImputer
        )  # pca1 | SimpleImputer
        expected_pipeline = (pca1 | SimpleImputer) >> LogisticRegression
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

        replaced_choice = choice.replace(
            PCA, SimpleImputer
        )  # SimpleImputer | SimpleImputer
        expected_choice = SimpleImputer | SimpleImputer
        self.assertEqual(replaced_choice.to_json(), expected_choice.to_json())

        replaced_pipeline = pipeline_choice.replace(
            PCA, SimpleImputer
        )  # SimpleImputer | SimpleImputer
        expected_pipeline = (SimpleImputer | SimpleImputer) >> LogisticRegression
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

    def test_nested_choice(self):
        pca1 = PCA(n_components=10)
        pca2 = PCA(n_components=5)
        pipeline_nested_choice = pca1 >> (pca1 | pca2)

        replaced_pipeline = pipeline_nested_choice.replace(pca1, SimpleImputer)
        expected_pipeline = SimpleImputer >> (SimpleImputer | pca2)
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

        replaced_pipeline = pipeline_nested_choice.replace(PCA, SimpleImputer)
        expected_pipeline = SimpleImputer >> (SimpleImputer | SimpleImputer)
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

    def test_nested_pipeline(self):
        pca1 = PCA(n_components=10)
        pca2 = PCA(n_components=5)
        first_pipeline = pca1 >> LogisticRegression
        nested_pipeline = pca1 >> (pca2 | first_pipeline)

        replaced_pipeline = nested_pipeline.replace(pca1, SimpleImputer)
        expected_pipeline = SimpleImputer >> (
            pca2 | (SimpleImputer >> LogisticRegression)
        )
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

        replaced_pipeline = nested_pipeline.replace(PCA, SimpleImputer)
        expected_pipeline = SimpleImputer >> (
            SimpleImputer | (SimpleImputer >> LogisticRegression)
        )
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

    def test_hyperparam_estimator(self):
        lr = LogisticRegression()
        linear_reg = LinearRegression()
        ada = AdaBoostRegressor(base_estimator=lr)

        replaced_ada = ada.replace(lr, linear_reg)
        expected_ada = AdaBoostRegressor(base_estimator=linear_reg)
        self.assertEqual(replaced_ada.to_json(), expected_ada.to_json())

        replaced_ada = ada.replace(LogisticRegression, linear_reg)
        expected_ada = AdaBoostRegressor(base_estimator=linear_reg)
        self.assertEqual(replaced_ada.to_json(), expected_ada.to_json())

        ada_pipeline = PCA >> SimpleImputer >> ada
        replaced_pipeline = ada_pipeline.replace(lr, linear_reg)
        expected_pipeline = (
            PCA >> SimpleImputer >> AdaBoostRegressor(base_estimator=linear_reg)
        )
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

        ada_choice = PCA | ada
        replaced_choice = ada_choice.replace(lr, linear_reg)
        expected_choice = PCA | AdaBoostRegressor(base_estimator=linear_reg)
        self.assertEqual(replaced_choice.to_json(), expected_choice.to_json())

        rfe = RFE(estimator=lr)
        replaced_rfe = rfe.replace(lr, linear_reg)
        expected_rfe = RFE(estimator=linear_reg)
        self.assertEqual(replaced_rfe.to_json(), expected_rfe.to_json())

    def test_hyperparam_estimator_list(self):
        lr = LogisticRegression()
        linear_reg = LinearRegression()
        dtc = DecisionTreeClassifier()

        cls_list = [("lr", lr), ("linear_reg", linear_reg)]
        vc = VotingClassifier(estimators=cls_list)

        replaced_vc = vc.replace(linear_reg, dtc)
        new_cls_list = [("lr", lr), ("linear_reg", dtc)]
        expected_vc = VotingClassifier(estimators=new_cls_list)
        self.assertEqual(replaced_vc.to_json(), expected_vc.to_json())

        sc = StackingClassifier(estimators=cls_list, final_estimator=vc)
        replaced_sc = sc.replace(linear_reg, dtc)
        new_cls_list = [("lr", lr), ("linear_reg", dtc)]
        expected_sc = StackingClassifier(
            estimators=new_cls_list, final_estimator=expected_vc
        )
        self.assertEqual(replaced_sc.to_json(), expected_sc.to_json())

    def test_replace_choice(self):
        choice = PCA | SelectKBest
        choice_pipeline = choice >> LogisticRegression

        replaced_pipeline = choice_pipeline.replace(choice, SelectKBest)
        expected_pipeline = SelectKBest >> LogisticRegression
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())

        choice2 = NoOp | LinearRegression
        replaced_pipeline = choice_pipeline.replace(LogisticRegression, choice2)
        expected_pipeline = choice >> choice2
        self.assertEqual(replaced_pipeline.to_json(), expected_pipeline.to_json())
