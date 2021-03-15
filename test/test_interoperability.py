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
from test import EnableSchemaValidation

from jsonschema.exceptions import ValidationError

import lale.type_checking
from lale.lib.lale import ConcatFeatures, NoOp
from lale.lib.sklearn import PCA, LogisticRegression, Nystroem


@unittest.skip(
    "Skipping here because travis-ci fails to allocate memory. This runs on internal travis."
)
class TestResNet50(unittest.TestCase):
    def test_init_fit_predict(self):
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms

        from lale.lib.pytorch import ResNet50

        transform = transforms.Compose([transforms.ToTensor()])

        data_train = datasets.FakeData(
            size=50, num_classes=2, transform=transform
        )  # , target_transform = transform)
        clf = ResNet50(num_classes=2, num_epochs=1)
        clf.fit(data_train)
        _ = clf.predict(data_train)


class TestResamplers(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(
            n_classes=2,
            class_sep=2,
            weights=[0.1, 0.9],
            n_informative=3,
            n_redundant=1,
            flip_y=0,
            n_features=20,
            n_clusters_per_class=1,
            n_samples=1000,
            random_state=10,
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)


def create_function_test_resampler(res_name):
    def test_resampler(self):
        from lale.lib.sklearn import PCA, LogisticRegression

        X_train, y_train = self.X_train, self.y_train
        X_test = self.X_test
        import importlib

        module_name = ".".join(res_name.split(".")[0:-1])
        class_name = res_name.split(".")[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        with EnableSchemaValidation():
            with self.assertRaises(ValidationError):
                _ = class_()

        # test_schemas_are_schemas
        lale.type_checking.validate_is_schema(class_.input_schema_fit())
        lale.type_checking.validate_is_schema(class_.input_schema_predict())
        lale.type_checking.validate_is_schema(class_.output_schema_predict())
        lale.type_checking.validate_is_schema(class_.hyperparam_schema())

        # test_init_fit_predict
        from lale.operators import make_pipeline

        pipeline1 = PCA() >> class_(operator=make_pipeline(LogisticRegression()))
        trained = pipeline1.fit(X_train, y_train)
        _ = trained.predict(X_test)

        pipeline2 = class_(operator=make_pipeline(PCA(), LogisticRegression()))
        trained = pipeline2.fit(X_train, y_train)
        _ = trained.predict(X_test)

        # test_with_hyperopt
        from lale.lib.lale import Hyperopt

        optimizer = Hyperopt(
            estimator=PCA >> class_(operator=make_pipeline(LogisticRegression())),
            max_evals=1,
            show_progressbar=False,
        )
        trained_optimizer = optimizer.fit(X_train, y_train)
        _ = trained_optimizer.predict(X_test)

        pipeline3 = class_(
            operator=PCA()
            >> (Nystroem & NoOp)
            >> ConcatFeatures
            >> LogisticRegression()
        )
        optimizer = Hyperopt(estimator=pipeline3, max_evals=1, show_progressbar=False)
        trained_optimizer = optimizer.fit(X_train, y_train)
        _ = trained_optimizer.predict(X_test)

        pipeline4 = (
            (
                PCA >> class_(operator=make_pipeline(Nystroem()))
                & class_(operator=make_pipeline(Nystroem()))
            )
            >> ConcatFeatures
            >> LogisticRegression()
        )
        optimizer = Hyperopt(
            estimator=pipeline4, max_evals=1, scoring="roc_auc", show_progressbar=False
        )
        trained_optimizer = optimizer.fit(X_train, y_train)
        _ = trained_optimizer.predict(X_test)

        # test_cross_validation
        from lale.helpers import cross_val_score

        cv_results = cross_val_score(pipeline1, X_train, y_train, cv=2)
        self.assertEqual(len(cv_results), 2)

        # test_to_json
        pipeline1.to_json()

    test_resampler.__name__ = "test_{0}".format(res_name.split(".")[-1])
    return test_resampler


resamplers = [
    "lale.lib.imblearn.SMOTE",
    "lale.lib.imblearn.SMOTEENN",
    "lale.lib.imblearn.ADASYN",
    "lale.lib.imblearn.BorderlineSMOTE",
    "lale.lib.imblearn.SVMSMOTE",
    "lale.lib.imblearn.RandomOverSampler",
    "lale.lib.imblearn.CondensedNearestNeighbour",
    "lale.lib.imblearn.EditedNearestNeighbours",
    "lale.lib.imblearn.RepeatedEditedNearestNeighbours",
    "lale.lib.imblearn.AllKNN",
    "lale.lib.imblearn.InstanceHardnessThreshold",
]

for res in resamplers:
    setattr(
        TestResamplers,
        "test_{0}".format(res.split(".")[-1]),
        create_function_test_resampler(res),
    )


class TestImblearn(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(
            n_classes=2,
            class_sep=2,
            weights=[0.1, 0.9],
            n_informative=3,
            n_redundant=1,
            flip_y=0,
            n_features=20,
            n_clusters_per_class=1,
            n_samples=1000,
            random_state=10,
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_decision_function(self):
        from lale.lib.imblearn import SMOTE
        from lale.lib.sklearn import RandomForestClassifier
        from lale.operators import make_pipeline

        smote = SMOTE(operator=make_pipeline(RandomForestClassifier()))
        trained = smote.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)
        with self.assertRaises(AttributeError):
            trained.decision_function(self.X_test)

    def test_string_labels(self):
        from lale.lib.imblearn import CondensedNearestNeighbour

        print(type(CondensedNearestNeighbour))
        from lale.operators import make_pipeline

        y_train = ["low" if label == 0 else "high" for label in self.y_train]
        pipeline = CondensedNearestNeighbour(
            operator=make_pipeline(PCA(), LogisticRegression()),
            sampling_strategy=["high"],
        )
        trained = pipeline.fit(self.X_train, y_train)
        _ = trained.predict(self.X_test)
