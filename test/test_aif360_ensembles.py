# Copyright 2020 IBM Corporation
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
from typing import Any, Dict

try:
    import tensorflow as tf

    tensorflow_installed = True
except ImportError:
    tensorflow_installed = False

from lale.lib.aif360 import (
    CalibratedEqOddsPostprocessing,
    DisparateImpactRemover,
    PrejudiceRemover,
    fair_stratified_train_test_split,
)
from lale.lib.aif360.adversarial_debiasing import AdversarialDebiasing
from lale.lib.aif360.datasets import fetch_creditg_df
from lale.lib.sklearn import (
    AdaBoostClassifier,
    BaggingClassifier,
    DecisionTreeClassifier,
    LogisticRegression,
    StackingClassifier,
    VotingClassifier,
)


class TestEnsemblesWithAIF360(unittest.TestCase):
    train_X = None
    train_y = None
    test_X = None
    fairness_info: Dict[str, Any] = {"temp": 0}

    @classmethod
    def setUpClass(cls) -> None:
        X, y, fi = fetch_creditg_df(preprocess=True)
        train_X, test_X, train_y, _ = fair_stratified_train_test_split(X, y, **fi)
        cls.train_X = train_X
        cls.train_y = train_y
        cls.test_X = test_X
        cls.fairness_info = fi

    @classmethod
    def _attempt_fit_predict(cls, model):
        trained = model.fit(cls.train_X, cls.train_y)
        trained.predict(cls.test_X)

    def test_bagging_pre_estimator_mitigation_ensemble(self):
        model = DisparateImpactRemover(**self.fairness_info) >> BaggingClassifier(
            base_estimator=DecisionTreeClassifier()
        )
        self._attempt_fit_predict(model)

    def test_bagging_post_estimator_mitigation_ensemble(self):
        model = CalibratedEqOddsPostprocessing(
            **self.fairness_info,
            estimator=BaggingClassifier(base_estimator=DecisionTreeClassifier())
        )
        self._attempt_fit_predict(model)

    def test_bagging_pre_estimator_mitigation_base(self):
        model = BaggingClassifier(
            base_estimator=DisparateImpactRemover(**self.fairness_info)
            >> DecisionTreeClassifier()
        )
        self._attempt_fit_predict(model)

    def test_bagging_in_estimator_mitigation_base(self):
        model = BaggingClassifier(base_estimator=PrejudiceRemover(**self.fairness_info))
        self._attempt_fit_predict(model)

    def test_bagging_in_estimator_mitigation_base_1(self):
        if tensorflow_installed:
            tf.compat.v1.disable_eager_execution()
            model = BaggingClassifier(
                base_estimator=AdversarialDebiasing(**self.fairness_info),
                n_estimators=2,
            )
            self._attempt_fit_predict(model)

    def test_bagging_post_estimator_mitigation_base(self):
        model = BaggingClassifier(
            base_estimator=CalibratedEqOddsPostprocessing(
                **self.fairness_info, estimator=DecisionTreeClassifier()
            )
        )
        self._attempt_fit_predict(model)

    def test_adaboost_pre_estimator_mitigation_ensemble(self):
        model = DisparateImpactRemover(**self.fairness_info) >> AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier()
        )
        self._attempt_fit_predict(model)

    def test_adaboost_post_estimator_mitigation_ensemble(self):
        model = CalibratedEqOddsPostprocessing(
            **self.fairness_info,
            estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        )
        self._attempt_fit_predict(model)

    def test_adaboost_pre_estimator_mitigation_base(self):
        model = AdaBoostClassifier(
            base_estimator=DisparateImpactRemover(**self.fairness_info)
            >> DecisionTreeClassifier()
        )
        self._attempt_fit_predict(model)

    def test_adaboost_in_estimator_mitigation_base(self):
        model = AdaBoostClassifier(
            base_estimator=PrejudiceRemover(**self.fairness_info)
        )
        self._attempt_fit_predict(model)

    def test_adaboost_post_estimator_mitigation_base(self):
        model = AdaBoostClassifier(
            base_estimator=CalibratedEqOddsPostprocessing(
                **self.fairness_info, estimator=DecisionTreeClassifier()
            )
        )
        self._attempt_fit_predict(model)

    def test_voting_pre_estimator_mitigation_ensemble(self):
        model = DisparateImpactRemover(**self.fairness_info) >> VotingClassifier(
            estimators=[("dtc", DecisionTreeClassifier()), ("lr", LogisticRegression())]
        )
        self._attempt_fit_predict(model)

    @unittest.skip("TODO: find out why it does not find predict_proba")
    def test_voting_post_estimator_mitigation_ensemble(self):
        model = CalibratedEqOddsPostprocessing(
            **self.fairness_info,
            estimator=VotingClassifier(
                estimators=[
                    ("dtc", DecisionTreeClassifier()),
                    ("lr", LogisticRegression()),
                ]
            )
        )
        self._attempt_fit_predict(model)

    def test_voting_pre_estimator_mitigation_base(self):
        model = VotingClassifier(
            estimators=[
                (
                    "dir+dtc",
                    DisparateImpactRemover(**self.fairness_info)
                    >> DecisionTreeClassifier(),
                ),
                ("lr", LogisticRegression()),
            ]
        )
        self._attempt_fit_predict(model)

    def test_voting_in_estimator_mitigation_base(self):
        model = VotingClassifier(
            estimators=[
                ("pr", PrejudiceRemover(**self.fairness_info)),
                ("lr", LogisticRegression()),
            ]
        )
        self._attempt_fit_predict(model)

    def test_voting_post_estimator_mitigation_base(self):
        model = VotingClassifier(
            estimators=[
                (
                    "dtc+ceop",
                    CalibratedEqOddsPostprocessing(
                        **self.fairness_info, estimator=DecisionTreeClassifier()
                    ),
                ),
                ("lr", LogisticRegression()),
            ]
        )
        self._attempt_fit_predict(model)

    def test_stacking_pre_estimator_mitigation_ensemble(self):
        model = DisparateImpactRemover(**self.fairness_info) >> StackingClassifier(
            estimators=[("dtc", DecisionTreeClassifier()), ("lr", LogisticRegression())]
        )
        self._attempt_fit_predict(model)

    def test_stacking_post_estimator_mitigation_ensemble(self):
        model = CalibratedEqOddsPostprocessing(
            **self.fairness_info,
            estimator=StackingClassifier(
                estimators=[
                    ("dtc", DecisionTreeClassifier()),
                    ("lr", LogisticRegression()),
                ]
            )
        )
        self._attempt_fit_predict(model)

    def test_stacking_pre_estimator_mitigation_base_only(self):
        model = StackingClassifier(
            estimators=[
                (
                    "dir+dtc",
                    DisparateImpactRemover(**self.fairness_info)
                    >> DecisionTreeClassifier(),
                ),
                ("lr", LogisticRegression()),
            ]
        )
        self._attempt_fit_predict(model)

    def test_stacking_pre_estimator_mitigation_base_and_final(self):
        model = StackingClassifier(
            estimators=[
                (
                    "dir+dtc",
                    DisparateImpactRemover(**self.fairness_info)
                    >> DecisionTreeClassifier(),
                ),
                ("lr", LogisticRegression()),
            ],
            final_estimator=DisparateImpactRemover(**self.fairness_info)
            >> DecisionTreeClassifier(),
            passthrough=True,
        )
        self._attempt_fit_predict(model)

    def test_stacking_pre_estimator_mitigation_final_only(self):
        model = StackingClassifier(
            estimators=[
                ("dtc", DecisionTreeClassifier()),
                ("lr", LogisticRegression()),
            ],
            final_estimator=DisparateImpactRemover(**self.fairness_info)
            >> DecisionTreeClassifier(),
            passthrough=True,
        )
        self._attempt_fit_predict(model)

    def test_stacking_in_estimator_mitigation_base_only(self):
        model = StackingClassifier(
            estimators=[
                ("pr", PrejudiceRemover(**self.fairness_info)),
                ("lr", LogisticRegression()),
            ]
        )
        self._attempt_fit_predict(model)

    def test_stacking_in_estimator_mitigation_base_and_final(self):
        model = StackingClassifier(
            estimators=[
                ("pr", PrejudiceRemover(**self.fairness_info)),
                ("lr", LogisticRegression()),
            ],
            final_estimator=PrejudiceRemover(**self.fairness_info),
            passthrough=True,
        )
        self._attempt_fit_predict(model)

    def test_stacking_in_estimator_mitigation_final_only(self):
        model = StackingClassifier(
            estimators=[
                ("dtc", DecisionTreeClassifier()),
                ("lr", LogisticRegression()),
            ],
            final_estimator=PrejudiceRemover(**self.fairness_info),
            passthrough=True,
        )
        self._attempt_fit_predict(model)

    def test_stacking_post_estimator_mitigation_base_only(self):
        model = StackingClassifier(
            estimators=[
                (
                    "dtc+ceop",
                    CalibratedEqOddsPostprocessing(
                        **self.fairness_info, estimator=DecisionTreeClassifier()
                    ),
                ),
                ("lr", LogisticRegression()),
            ]
        )
        self._attempt_fit_predict(model)

    def test_stacking_post_estimator_mitigation_base_and_final(self):
        model = StackingClassifier(
            estimators=[
                (
                    "dtc+ceop",
                    CalibratedEqOddsPostprocessing(
                        **self.fairness_info, estimator=DecisionTreeClassifier()
                    ),
                ),
                ("lr", LogisticRegression()),
            ],
            final_estimator=CalibratedEqOddsPostprocessing(
                **self.fairness_info, estimator=DecisionTreeClassifier()
            ),
            passthrough=True,
        )
        self._attempt_fit_predict(model)

    def test_stacking_post_estimator_mitigation_final_only(self):
        model = StackingClassifier(
            estimators=[
                ("dtc", DecisionTreeClassifier()),
                ("lr", LogisticRegression()),
            ],
            final_estimator=CalibratedEqOddsPostprocessing(
                **self.fairness_info, estimator=DecisionTreeClassifier()
            ),
            passthrough=True,
        )
        self._attempt_fit_predict(model)
