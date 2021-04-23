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

import numpy as np
import sklearn.datasets
import sklearn.model_selection

import lale.lib.autoai_libs
from lale.lib.autoai_libs import float32_transform
from lale.lib.lale import Hyperopt
from lale.lib.sklearn import LogisticRegression as LR
from lale.lib.xgboost.xgb_classifier import XGBClassifier


class TestAutoaiLibs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        iris = sklearn.datasets.load_iris()
        iris_X, iris_y = iris.data, iris.target
        (
            iris_train_X,
            iris_test_X,
            iris_train_y,
            iris_test_y,
        ) = sklearn.model_selection.train_test_split(iris_X, iris_y)
        cls._iris = {
            "train_X": iris_train_X,
            "train_y": iris_train_y,
            "test_X": iris_test_X,
            "test_y": iris_test_y,
        }

    def doTest(self, trainable, train_X, train_y, test_X, test_y):
        trained = trainable.fit(train_X, train_y)
        _ = trained.transform(test_X)
        with self.assertWarns(DeprecationWarning):
            trainable.transform(train_X)
        trainable.to_json()
        trainable_pipeline = trainable >> float32_transform() >> LR()
        trained_pipeline = trainable_pipeline.fit(train_X, train_y)
        trained_pipeline.predict(test_X)
        hyperopt = Hyperopt(estimator=trainable_pipeline, max_evals=1, verbose=True)
        trained_hyperopt = hyperopt.fit(train_X, train_y)
        trained_hyperopt.predict(test_X)

    def test_NumpyColumnSelector(self):
        trainable = lale.lib.autoai_libs.NumpyColumnSelector()
        self.doTest(trainable, **self._iris)

    def test_CompressStrings(self):
        n_columns = self._iris["train_X"].shape[1]
        trainable = lale.lib.autoai_libs.CompressStrings(
            dtypes_list=["int_num" for i in range(n_columns)],
            misslist_list=[[] for i in range(n_columns)],
        )
        self.doTest(trainable, **self._iris)

    def test_NumpyReplaceMissingValues(self):
        trainable = lale.lib.autoai_libs.NumpyReplaceMissingValues()
        self.doTest(trainable, **self._iris)

    def test_NumpyReplaceUnknownValues(self):
        trainable = lale.lib.autoai_libs.NumpyReplaceUnknownValues(filling_values=42.0)
        self.doTest(trainable, **self._iris)

    def test_boolean2float(self):
        trainable = lale.lib.autoai_libs.boolean2float()
        self.doTest(trainable, **self._iris)

    def test_CatImputer(self):
        trainable = lale.lib.autoai_libs.CatImputer()
        self.doTest(trainable, **self._iris)

    def test_CatEncoder(self):
        trainable = lale.lib.autoai_libs.CatEncoder(
            encoding="ordinal",
            categories="auto",
            dtype="float64",
            handle_unknown="ignore",
        )
        self.doTest(trainable, **self._iris)

    def test_float32_transform(self):
        trainable = lale.lib.autoai_libs.float32_transform()
        self.doTest(trainable, **self._iris)

    def test_FloatStr2Float(self):
        n_columns = self._iris["train_X"].shape[1]
        trainable = lale.lib.autoai_libs.FloatStr2Float(
            dtypes_list=["int_num" for i in range(n_columns)]
        )
        self.doTest(trainable, **self._iris)

    def test_OptStandardScaler(self):
        trainable = lale.lib.autoai_libs.OptStandardScaler()
        self.doTest(trainable, **self._iris)

    def test_NumImputer(self):
        trainable = lale.lib.autoai_libs.NumImputer()
        self.doTest(trainable, **self._iris)

    def test_NumpyPermuteArray(self):
        trainable = lale.lib.autoai_libs.NumpyPermuteArray(
            axis=0, permutation_indices=[2, 0, 1, 3]
        )
        self.doTest(trainable, **self._iris)

    def test_TNoOp(self):
        from autoai_libs.utils.fc_methods import is_not_categorical

        trainable = lale.lib.autoai_libs.TNoOp(
            fun=np.rint,
            name="do nothing",
            datatypes=["numeric"],
            feat_constraints=[is_not_categorical],
        )
        self.doTest(trainable, **self._iris)

    def test_TA1(self):
        from autoai_libs.utils.fc_methods import is_not_categorical

        float32 = np.dtype("float32")
        trainable = lale.lib.autoai_libs.TA1(
            fun=np.rint,
            name="round",
            datatypes=["numeric"],
            feat_constraints=[is_not_categorical],
            col_names=["a", "b", "c", "d"],
            col_dtypes=[float32, float32, float32, float32],
        )
        self.doTest(trainable, **self._iris)

    def test_TA2(self):
        from autoai_libs.utils.fc_methods import is_not_categorical

        float32 = np.dtype("float32")
        trainable = lale.lib.autoai_libs.TA2(
            fun=np.add,
            name="sum",
            datatypes1=["numeric"],
            feat_constraints1=[is_not_categorical],
            datatypes2=["numeric"],
            feat_constraints2=[is_not_categorical],
            col_names=["a", "b", "c", "d"],
            col_dtypes=[float32, float32, float32, float32],
        )
        self.doTest(trainable, **self._iris)

    def test_TB1(self):
        from autoai_libs.utils.fc_methods import is_not_categorical
        from sklearn.preprocessing import StandardScaler

        float32 = np.dtype("float32")
        trainable = lale.lib.autoai_libs.TB1(
            tans_class=StandardScaler,
            name="stdscaler",
            datatypes=["numeric"],
            feat_constraints=[is_not_categorical],
            col_names=["a", "b", "c", "d"],
            col_dtypes=[float32, float32, float32, float32],
        )
        self.doTest(trainable, **self._iris)

    def test_TB2(self):
        pass  # TODO: not sure how to instantiate, what to pass for tans_class

    def test_TAM(self):
        from autoai_libs.cognito.transforms.transform_extras import (
            IsolationForestAnomaly,
        )

        float32 = np.dtype("float32")
        trainable = lale.lib.autoai_libs.TAM(
            tans_class=IsolationForestAnomaly,
            name="isoforestanomaly",
            col_names=["a", "b", "c", "d"],
            col_dtypes=[float32, float32, float32, float32],
        )
        self.doTest(trainable, **self._iris)

    def test_TGen(self):
        from autoai_libs.cognito.transforms.transform_extras import NXOR
        from autoai_libs.utils.fc_methods import is_not_categorical

        float32 = np.dtype("float32")
        trainable = lale.lib.autoai_libs.TGen(
            fun=NXOR,
            name="nxor",
            arg_count=2,
            datatypes_list=[["numeric"], ["numeric"]],
            feat_constraints_list=[[is_not_categorical], [is_not_categorical]],
            col_names=["a", "b", "c", "d"],
            col_dtypes=[float32, float32, float32, float32],
        )
        self.doTest(trainable, **self._iris)

    def test_FS1(self):
        trainable = lale.lib.autoai_libs.FS1(
            cols_ids_must_keep=[1],
            additional_col_count_to_keep=3,
            ptype="classification",
        )
        self.doTest(trainable, **self._iris)

    def test_FS2(self):
        from sklearn.ensemble import ExtraTreesClassifier

        trainable = lale.lib.autoai_libs.FS2(
            cols_ids_must_keep=[1],
            additional_col_count_to_keep=3,
            ptype="classification",
            eval_algo=ExtraTreesClassifier,
        )
        self.doTest(trainable, **self._iris)


class TestAutoaiLibsText(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import fetch_20newsgroups

        cats = ["alt.atheism", "sci.space"]
        newsgroups_train = fetch_20newsgroups(subset="train", categories=cats)
        self.train_X, self.train_y = (
            np.array(newsgroups_train.data),
            newsgroups_train.target,
        )
        self.train_X = np.reshape(self.train_X, (self.train_X.shape[0], 1))
        newsgroups_test = fetch_20newsgroups(subset="test", categories=cats)
        self.test_X, self.test_y = (
            np.array(newsgroups_test.data),
            newsgroups_test.target,
        )
        self.test_X = np.reshape(self.test_X, (self.test_X.shape[0], 1))

    def doTest(self, trainable, train_X, train_y, test_X, test_y):
        trained = trainable.fit(train_X, train_y)
        _ = trained.transform(test_X)
        with self.assertWarns(DeprecationWarning):
            trainable.transform(train_X)
        trainable.to_json()
        trainable_pipeline = trainable >> float32_transform() >> XGBClassifier()
        trained_pipeline = trainable_pipeline.fit(train_X, train_y)
        trained_pipeline.predict(test_X)
        hyperopt = Hyperopt(estimator=trainable_pipeline, max_evals=1, verbose=True)
        trained_hyperopt = hyperopt.fit(train_X, train_y)
        trained_hyperopt.predict(test_X)

    def test_TextTransformer(self):
        trainable = lale.lib.autoai_libs.TextTransformer(
            drop_columns=True,
            columns_to_be_deleted=[0, 1],
            text_processing_options={"word2vec": {"output_dim": 5}},
        )
        self.doTest(trainable, self.train_X, self.train_y, self.test_X, self.test_y)

    def test_Word2VecTransformer(self):
        trainable = lale.lib.autoai_libs.Word2VecTransformer(
            drop_columns=True, output_dim=5
        )
        self.doTest(trainable, self.train_X, self.train_y, self.test_X, self.test_y)
