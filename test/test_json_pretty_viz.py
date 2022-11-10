# Copyright 2019-2022 IBM Corporation
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

import lale.operators
import lale.pretty_print


class TestToGraphviz(unittest.TestCase):
    def test_with_operator_choice(self):
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import (
            PCA,
            KNeighborsClassifier,
            LogisticRegression,
            Nystroem,
        )
        from lale.operators import make_choice

        kernel_tfm_or_not = NoOp | Nystroem
        tfm = PCA
        clf = make_choice(LogisticRegression, KNeighborsClassifier)
        clf.visualize(ipython_display=False)
        optimizable = kernel_tfm_or_not >> tfm >> clf
        optimizable.visualize(ipython_display=False)

    def test_invalid_input(self):
        from sklearn.linear_model import LogisticRegression as SklearnLR

        scikit_lr = SklearnLR()
        from lale.helpers import to_graphviz

        with self.assertRaises(TypeError):
            to_graphviz(scikit_lr)


class TestPrettyPrint(unittest.TestCase):
    def _roundtrip(self, expected, printed):
        self.maxDiff = None
        # sklearn_version_family changes based on the Python as well as sklearn version,
        # so remove that hyperparameter while comparing if present.
        import re

        expected = re.sub(r"""sklearn_version_family=.\d*.,""", "", expected)
        printed = re.sub(r"""sklearn_version_family=.\d*.,""", "", printed)
        self.assertEqual(expected, printed)
        globals2 = {}
        locals2 = {}
        try:
            exec(printed, globals2, locals2)
        except Exception as e:
            import pprint

            print("error during exec(printed, globals2, locals2) where:")
            print(f'printed = """{printed}"""')
            print(f"globals2 = {pprint.pformat(globals2)}")
            print(f"locals2 = {pprint.pformat(locals2)}")
            raise e
        pipeline2 = locals2["pipeline"]
        import sklearn.pipeline

        self.assertIsInstance(
            pipeline2, (lale.operators.PlannedOperator, sklearn.pipeline.Pipeline)
        )

    def test_distance_threshold_validation_error(self):
        import jsonschema

        from lale.lib.sklearn import FeatureAgglomeration, LogisticRegression

        with self.assertRaises(jsonschema.ValidationError):
            _ = (
                FeatureAgglomeration(
                    distance_threshold=0.5, n_clusters=None, compute_full_tree=True
                )
                >> LogisticRegression()
            )

    def test_indiv_op_1(self):
        from lale.lib.sklearn import LogisticRegression

        pipeline = LogisticRegression(solver=LogisticRegression.enum.solver.saga, C=0.9)
        expected = """from sklearn.linear_model import LogisticRegression
import lale

lale.wrap_imported_operators()
pipeline = LogisticRegression(solver="saga", C=0.9)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_indiv_op_2(self):
        from lale.lib.sklearn import LogisticRegression

        pipeline = LogisticRegression()
        expected = """from sklearn.linear_model import LogisticRegression
import lale

lale.wrap_imported_operators()
pipeline = LogisticRegression()"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_reducible(self):
        from lale.lib.lale import NoOp
        from lale.lib.rasl import ConcatFeatures
        from lale.lib.sklearn import (
            PCA,
            KNeighborsClassifier,
            LogisticRegression,
            MinMaxScaler,
            Nystroem,
        )
        from lale.lib.xgboost import XGBClassifier as XGB

        pca = PCA(copy=False)
        logistic_regression = LogisticRegression(solver="saga", C=0.9)
        pipeline = (
            (MinMaxScaler | NoOp)
            >> (pca & Nystroem)
            >> ConcatFeatures
            >> (KNeighborsClassifier | logistic_regression | XGB)
        )
        expected = """from sklearn.preprocessing import MinMaxScaler
from lale.lib.lale import NoOp
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from lale.lib.rasl import ConcatFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier as XGB
import lale

lale.wrap_imported_operators()
pca = PCA(copy=False)
logistic_regression = LogisticRegression(solver="saga", C=0.9)
pipeline = (
    (MinMaxScaler | NoOp)
    >> (pca & Nystroem)
    >> ConcatFeatures
    >> (KNeighborsClassifier | logistic_regression | XGB)
)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_no_combinators(self):
        from lale.lib.lale import NoOp
        from lale.lib.rasl import ConcatFeatures
        from lale.lib.sklearn import (
            PCA,
            KNeighborsClassifier,
            LogisticRegression,
            MinMaxScaler,
            Nystroem,
        )

        pca = PCA(copy=False)
        logistic_regression = LogisticRegression(solver="saga", C=0.9)
        pipeline = (
            (MinMaxScaler | NoOp)
            >> (pca & Nystroem & NoOp)
            >> ConcatFeatures
            >> (KNeighborsClassifier | logistic_regression)
        )
        expected = """from sklearn.preprocessing import MinMaxScaler
from lale.lib.lale import NoOp
from lale.operators import make_choice
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from lale.operators import make_union
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from lale.operators import make_pipeline

choice_0 = make_choice(MinMaxScaler, NoOp)
pca = PCA(copy=False)
union = make_union(pca, Nystroem, NoOp)
logistic_regression = LogisticRegression(solver="saga", C=0.9)
choice_1 = make_choice(KNeighborsClassifier, logistic_regression)
pipeline = make_pipeline(choice_0, union, choice_1)"""
        printed = lale.pretty_print.to_string(pipeline, combinators=False)
        self._roundtrip(expected, printed)

    def test_astype_sklearn(self):
        from lale.lib.rasl import ConcatFeatures
        from lale.lib.sklearn import PCA, LogisticRegression, MinMaxScaler, Nystroem

        pca = PCA(copy=False)
        logistic_regression = LogisticRegression(solver="saga", C=0.9)
        pipeline = (
            MinMaxScaler()
            >> (pca & Nystroem())
            >> ConcatFeatures
            >> logistic_regression
        )
        expected = """from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_union
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pca = PCA(copy=False)
union = make_union(pca, Nystroem())
logistic_regression = LogisticRegression(solver="saga", C=0.9)
pipeline = make_pipeline(MinMaxScaler(), union, logistic_regression)"""
        printed = lale.pretty_print.to_string(pipeline, astype="sklearn")
        self._roundtrip(expected, printed)

    def test_import_as_1(self):
        from lale.lib.sklearn import LogisticRegression as LR

        pipeline = LR(solver="saga", C=0.9)
        expected = """from sklearn.linear_model import LogisticRegression as LR
import lale

lale.wrap_imported_operators()
pipeline = LR(solver="saga", C=0.9)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_import_as_2(self):
        from lale.lib.lale import NoOp
        from lale.lib.rasl import ConcatFeatures as Concat
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.lib.sklearn import MinMaxScaler as Scaler
        from lale.lib.sklearn import Nystroem

        pca = PCA(copy=False)
        lr = LR(solver="saga", C=0.9)
        pipeline = (Scaler | NoOp) >> (pca & Nystroem) >> Concat >> (KNN | lr)
        expected = """from sklearn.preprocessing import MinMaxScaler as Scaler
from lale.lib.lale import NoOp
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from lale.lib.rasl import ConcatFeatures as Concat
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
import lale

lale.wrap_imported_operators()
pca = PCA(copy=False)
lr = LR(solver="saga", C=0.9)
pipeline = (Scaler | NoOp) >> (pca & Nystroem) >> Concat >> (KNN | lr)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_operator_choice(self):
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import MinMaxScaler as Scl

        pipeline = PCA | Scl
        expected = """from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler as Scl
import lale

lale.wrap_imported_operators()
pipeline = PCA | Scl"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_higher_order(self):
        from lale.lib.lale import Both
        from lale.lib.sklearn import PCA, Nystroem

        pipeline = Both(op1=PCA(n_components=2), op2=Nystroem)
        expected = """from lale.lib.lale import Both
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
import lale

lale.wrap_imported_operators()
pca = PCA(n_components=2)
pipeline = Both(op1=pca, op2=Nystroem)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_higher_order_2(self):
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.lib.sklearn import VotingClassifier as Vote

        pipeline = Vote(
            estimators=[("knn", KNN), ("pipeline", PCA() >> LR)], voting="soft"
        )
        expected = """from sklearn.ensemble import VotingClassifier as Vote
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as LR
import lale

lale.wrap_imported_operators()
pipeline = Vote(
    estimators=[("knn", KNN), ("pipeline", PCA() >> LR)], voting="soft"
)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_multimodal(self):
        from lale.lib.rasl import ConcatFeatures as Cat
        from lale.lib.rasl import Project
        from lale.lib.sklearn import LinearSVC
        from lale.lib.sklearn import Normalizer as Norm
        from lale.lib.sklearn import OneHotEncoder as OneHot

        project_0 = Project(columns={"type": "number"})
        project_1 = Project(columns={"type": "string"})
        linear_svc = LinearSVC(C=29617.4, dual=False, tol=0.005266)
        pipeline = (
            ((project_0 >> Norm()) & (project_1 >> OneHot())) >> Cat >> linear_svc
        )
        expected = """from lale.lib.rasl import Project
from sklearn.preprocessing import Normalizer as Norm
from sklearn.preprocessing import OneHotEncoder as OneHot
from lale.lib.rasl import ConcatFeatures as Cat
from sklearn.svm import LinearSVC
import lale

lale.wrap_imported_operators()
project_0 = Project(columns={"type": "number"})
project_1 = Project(columns={"type": "string"})
linear_svc = LinearSVC(C=29617.4, dual=False, tol=0.005266)
pipeline = (
    ((project_0 >> Norm()) & (project_1 >> OneHot())) >> Cat >> linear_svc
)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_irreducible_1(self):
        from lale.lib.sklearn import (
            PCA,
            KNeighborsClassifier,
            LogisticRegression,
            MinMaxScaler,
            Nystroem,
        )
        from lale.operators import make_pipeline_graph

        choice = PCA | Nystroem
        pipeline = make_pipeline_graph(
            steps=[choice, MinMaxScaler, LogisticRegression, KNeighborsClassifier],
            edges=[
                (choice, LogisticRegression),
                (MinMaxScaler, LogisticRegression),
                (MinMaxScaler, KNeighborsClassifier),
            ],
        )
        expected = """from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lale.operators import make_pipeline_graph
import lale

lale.wrap_imported_operators()
choice = PCA | Nystroem
pipeline = make_pipeline_graph(
    steps=[choice, MinMaxScaler, LogisticRegression, KNeighborsClassifier],
    edges=[
        (choice, LogisticRegression),
        (MinMaxScaler, LogisticRegression),
        (MinMaxScaler, KNeighborsClassifier),
    ],
)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_irreducible_2(self):
        from lale.lib.rasl import ConcatFeatures as HStack
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.lib.sklearn import MinMaxScaler as MMS
        from lale.operators import make_pipeline_graph

        pipeline_0 = HStack >> LR
        pipeline = make_pipeline_graph(
            steps=[PCA, MMS, KNN, pipeline_0],
            edges=[(PCA, KNN), (PCA, pipeline_0), (MMS, pipeline_0)],
        )
        expected = """from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.neighbors import KNeighborsClassifier as KNN
from lale.lib.rasl import ConcatFeatures as HStack
from sklearn.linear_model import LogisticRegression as LR
from lale.operators import make_pipeline_graph
import lale

lale.wrap_imported_operators()
pipeline_0 = HStack >> LR
pipeline = make_pipeline_graph(
    steps=[PCA, MMS, KNN, pipeline_0],
    edges=[(PCA, KNN), (PCA, pipeline_0), (MMS, pipeline_0)],
)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_nested(self):
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import LogisticRegression as LR

        lr_0 = LR(C=0.09)
        lr_1 = LR(C=0.19)
        pipeline = PCA >> (lr_0 | NoOp >> lr_1)
        expected = """from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as LR
from lale.lib.lale import NoOp
import lale

lale.wrap_imported_operators()
lr_0 = LR(C=0.09)
lr_1 = LR(C=0.19)
pipeline = PCA >> (lr_0 | NoOp >> lr_1)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_cat_encoder(self):
        import numpy as np
        from autoai_libs.transformers.exportable import CatEncoder

        from lale.lib.sklearn import LogisticRegression as LR

        cat_encoder = CatEncoder(
            encoding="ordinal",
            categories="auto",
            dtype=np.float64,
            handle_unknown="error",
        )
        pipeline = cat_encoder >> LR()
        expected = """from autoai_libs.transformers.exportable import CatEncoder
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
import lale

lale.wrap_imported_operators()
cat_encoder = CatEncoder(
    encoding="ordinal",
    categories="auto",
    dtype=np.float64,
    handle_unknown="error",
    sklearn_version_family="23",
)
pipeline = cat_encoder >> LR()"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_fs1(self):
        from autoai_libs.cognito.transforms.transform_utils import FS1

        from lale.lib.sklearn import LogisticRegression as LR

        fs1 = FS1(
            cols_ids_must_keep=range(0, 7),
            additional_col_count_to_keep=8,
            ptype="classification",
        )
        pipeline = fs1 >> LR()
        expected = """from autoai_libs.cognito.transforms.transform_utils import FS1
from sklearn.linear_model import LogisticRegression as LR
import lale

lale.wrap_imported_operators()
fs1 = FS1(
    cols_ids_must_keep=range(0, 7),
    additional_col_count_to_keep=8,
    ptype="classification",
)
pipeline = fs1 >> LR()"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_numpy_replace_missing_values(self):
        from autoai_libs.transformers.exportable import NumpyReplaceMissingValues

        from lale.lib.sklearn import LogisticRegression as LR

        numpy_replace_missing_values = NumpyReplaceMissingValues(
            filling_values=float("nan"), missing_values=["?"]
        )
        pipeline = numpy_replace_missing_values >> LR()
        expected = """from autoai_libs.transformers.exportable import NumpyReplaceMissingValues
from sklearn.linear_model import LogisticRegression as LR
import lale

lale.wrap_imported_operators()
numpy_replace_missing_values = NumpyReplaceMissingValues(
    missing_values=["?"], filling_values=float("nan")
)
pipeline = numpy_replace_missing_values >> LR()"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_numpy_replace_unknown_values1(self):
        from autoai_libs.transformers.exportable import NumpyReplaceUnknownValues

        from lale.lib.sklearn import LogisticRegression as LR

        numpy_replace_unknown_values = NumpyReplaceUnknownValues(
            filling_values=float("nan"),
            filling_values_list=[float("nan")],
            known_values_list=[[36, 45, 56, 67, 68, 75, 78, 89]],
            missing_values_reference_list=["", "-", "?", float("nan")],
        )
        pipeline = numpy_replace_unknown_values >> LR()
        expected = """from autoai_libs.transformers.exportable import NumpyReplaceUnknownValues
from sklearn.linear_model import LogisticRegression as LR
import lale

lale.wrap_imported_operators()
numpy_replace_unknown_values = NumpyReplaceUnknownValues(
    filling_values=float("nan"),
    filling_values_list=[float("nan")],
    missing_values_reference_list=["", "-", "?", float("nan")],
)
pipeline = numpy_replace_unknown_values >> LR()"""
        try:
            self._roundtrip(expected, lale.pretty_print.to_string(pipeline))
        except BaseException:
            expected = """from autoai_libs.transformers.exportable import NumpyReplaceUnknownValues
    from sklearn.linear_model import LogisticRegression as LR
    import lale

    lale.wrap_imported_operators()
    numpy_replace_unknown_values = NumpyReplaceUnknownValues(
        filling_values=float("nan"),
        filling_values_list=[float("nan")],
        missing_values_reference_list=("", "-", "?", float("nan")),
    )
    pipeline = numpy_replace_unknown_values >> LR()"""
            self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_numpy_replace_unknown_values2(self):
        from lale.lib.autoai_libs import NumpyReplaceUnknownValues
        from lale.lib.sklearn import LogisticRegression as LR

        CustomOp = NumpyReplaceUnknownValues.customize_schema(
            known_values_list={
                "anyOf": [
                    {"type": "array", "items": {"laleType": "Any"}},
                    {"enum": [None]},
                ],
                "default": None,
            }
        )
        numpy_replace_unknown_values = CustomOp(
            filling_values=float("nan"),
            filling_values_list=[float("nan")],
            known_values_list=[[36, 45, 56, 67, 68, 75, 78, 89]],
            missing_values_reference_list=["", "-", "?", float("nan")],
        )
        pipeline = numpy_replace_unknown_values >> LR()
        expected = """from autoai_libs.transformers.exportable import (
    NumpyReplaceUnknownValues as CustomOp,
)
from sklearn.linear_model import LogisticRegression as LR
import lale

lale.wrap_imported_operators()
custom_op = CustomOp(
    filling_values=float("nan"),
    filling_values_list=[float("nan")],
    known_values_list=[[36, 45, 56, 67, 68, 75, 78, 89]],
    missing_values_reference_list=["", "-", "?", float("nan")],
)
pipeline = custom_op >> LR()"""
        try:
            self._roundtrip(expected, lale.pretty_print.to_string(pipeline))
        except BaseException:
            expected = """from autoai_libs.transformers.exportable import (
        NumpyReplaceUnknownValues as CustomOp,
    )
    from sklearn.linear_model import LogisticRegression as LR
    import lale

    lale.wrap_imported_operators()
    custom_op = CustomOp(
        filling_values=float("nan"),
        filling_values_list=[float("nan")],
        known_values_list=[[36, 45, 56, 67, 68, 75, 78, 89]],
        missing_values_reference_list=("", "-", "?", float("nan")),
    )
    pipeline = custom_op >> LR()"""
            self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_tam_1(self):
        import autoai_libs.cognito.transforms.transform_extras
        import numpy as np
        from autoai_libs.cognito.transforms.transform_utils import TAM

        from lale.lib.sklearn import LogisticRegression as LR

        tam = TAM(
            tans_class=autoai_libs.cognito.transforms.transform_extras.IsolationForestAnomaly,
            name="isoforestanomaly",
            col_names=["a", "b", "c"],
            col_dtypes=[np.dtype("float32"), np.dtype("float32"), np.dtype("float32")],
        )
        pipeline = tam >> LR()
        expected = """from autoai_libs.cognito.transforms.transform_utils import TAM
import autoai_libs.cognito.transforms.transform_extras
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.pipeline import make_pipeline

tam = TAM(
    tans_class=autoai_libs.cognito.transforms.transform_extras.IsolationForestAnomaly,
    name="isoforestanomaly",
    col_names=["a", "b", "c"],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
    ],
)
pipeline = make_pipeline(tam, LR())"""
        self._roundtrip(
            expected, lale.pretty_print.to_string(pipeline, astype="sklearn")
        )

    def test_autoai_libs_tam_2(self):
        import numpy as np
        from lightgbm import LGBMClassifier
        from sklearn.decomposition import PCA

        from lale.lib.autoai_libs import TAM
        from lale.operators import make_pipeline

        pca = PCA(copy=False)
        tam = TAM(
            tans_class=pca,
            name="pca",
            col_names=["a", "b", "c"],
            col_dtypes=[np.dtype("float32"), np.dtype("float32"), np.dtype("float32")],
        )
        lgbm_classifier = LGBMClassifier(class_weight="balanced", learning_rate=0.18)
        pipeline = make_pipeline(tam, lgbm_classifier)
        expected = """from autoai_libs.cognito.transforms.transform_utils import TAM
import sklearn.decomposition
import numpy as np
from lightgbm import LGBMClassifier
from lale.operators import make_pipeline

tam = TAM(
    tans_class=sklearn.decomposition.PCA(copy=False),
    name="pca",
    col_names=["a", "b", "c"],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
    ],
)
lgbm_classifier = LGBMClassifier(
    class_weight="balanced", learning_rate=0.18, n_estimators=100
)
pipeline = make_pipeline(tam, lgbm_classifier)"""
        self._roundtrip(
            expected, lale.pretty_print.to_string(pipeline, combinators=False)
        )

    def test_autoai_libs_tam_3(self):
        import autoai_libs.cognito.transforms.transform_utils
        import numpy as np
        import sklearn.cluster
        import sklearn.linear_model
        import sklearn.pipeline

        import lale.helpers
        import lale.operators
        import lale.pretty_print

        sklearn_pipeline = sklearn.pipeline.make_pipeline(
            autoai_libs.cognito.transforms.transform_utils.TAM(
                tans_class=sklearn.cluster.FeatureAgglomeration(
                    affinity="euclidean",
                    compute_full_tree="auto",
                    connectivity=None,
                    linkage="ward",
                    memory=None,
                    n_clusters=2,
                    pooling_func=np.mean,
                ),
                name="featureagglomeration",
                col_names=["a", "b", "c"],
                col_dtypes=[
                    np.dtype("float32"),
                    np.dtype("float32"),
                    np.dtype("float32"),
                ],
            ),
            sklearn.linear_model.LogisticRegression(
                solver="liblinear", multi_class="ovr"
            ),
        )
        pipeline = lale.helpers.import_from_sklearn_pipeline(sklearn_pipeline)
        expected = """from autoai_libs.cognito.transforms.transform_utils import TAM
from sklearn.cluster import FeatureAgglomeration
import numpy as np
from sklearn.linear_model import LogisticRegression
import lale

lale.wrap_imported_operators()
tam = TAM(
    tans_class=FeatureAgglomeration(),
    name="featureagglomeration",
    col_names=["a", "b", "c"],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
    ],
)
logistic_regression = LogisticRegression(
    multi_class="ovr", solver="liblinear"
)
pipeline = tam >> logistic_regression"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_tam_4(self):
        import autoai_libs.cognito.transforms.transform_utils
        import numpy as np
        import sklearn.decomposition
        import sklearn.linear_model
        import sklearn.pipeline

        import lale.helpers
        import lale.operators
        import lale.pretty_print

        sklearn_pipeline = sklearn.pipeline.make_pipeline(
            autoai_libs.cognito.transforms.transform_utils.TAM(
                tans_class=sklearn.decomposition.PCA(),
                name="pca",
                col_names=["a", "b", "c"],
                col_dtypes=[
                    np.dtype("float32"),
                    np.dtype("float32"),
                    np.dtype("float32"),
                ],
            ),
            sklearn.linear_model.LogisticRegression(
                solver="liblinear", multi_class="ovr"
            ),
        )
        pipeline = lale.helpers.import_from_sklearn_pipeline(
            sklearn_pipeline, fitted=False
        )
        assert isinstance(pipeline, lale.operators.TrainableOperator)

        expected = """from autoai_libs.cognito.transforms.transform_utils import TAM
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LogisticRegression
import lale

lale.wrap_imported_operators()
tam = TAM(
    tans_class=PCA(),
    name="pca",
    col_names=["a", "b", "c"],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
    ],
)
logistic_regression = LogisticRegression(
    multi_class="ovr", solver="liblinear"
)
pipeline = tam >> logistic_regression"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))
        import numpy as np
        import pandas as pd

        test = pd.DataFrame(
            np.random.randint(0, 100, size=(15, 3)),
            columns=["a", "b", "c"],
            dtype=np.dtype("float32"),
        )
        trained = pipeline.fit(
            test.to_numpy(), [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1]
        )
        trained.predict(test.to_numpy())

    def test_autoai_libs_ta1(self):
        import autoai_libs.utils.fc_methods
        import numpy as np
        from autoai_libs.cognito.transforms.transform_utils import TA1

        from lale.lib.sklearn import LogisticRegression as LR

        ta1 = TA1(
            fun=np.rint,
            name="round",
            datatypes=["numeric"],
            feat_constraints=[autoai_libs.utils.fc_methods.is_not_categorical],
            col_names=[
                "a____________",
                "b____________",
                "c____________",
                "d____________",
                "e____________",
            ],
            col_dtypes=[
                np.dtype("float32"),
                np.dtype("float32"),
                np.dtype("float32"),
                np.dtype("float32"),
                np.dtype("float32"),
            ],
        )
        pipeline = ta1 >> LR()
        expected = """from autoai_libs.cognito.transforms.transform_utils import TA1
import numpy as np
import autoai_libs.utils.fc_methods
from sklearn.linear_model import LogisticRegression as LR
import lale

lale.wrap_imported_operators()
ta1 = TA1(
    fun=np.rint,
    name="round",
    datatypes=["numeric"],
    feat_constraints=[autoai_libs.utils.fc_methods.is_not_categorical],
    col_names=[
        "a____________", "b____________", "c____________", "d____________",
        "e____________",
    ],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"),
    ],
)
pipeline = ta1 >> LR()"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_t_no_op(self):
        from lightgbm import LGBMClassifier

        from lale.lib.autoai_libs import TNoOp
        from lale.operators import make_pipeline

        t_no_op = TNoOp(
            fun="fun",
            name="no_action",
            datatypes="x",
            feat_constraints=[],
            tgraph="tgraph",
        )
        lgbm_classifier = LGBMClassifier(class_weight="balanced", learning_rate=0.18)
        pipeline = make_pipeline(t_no_op, lgbm_classifier)
        expected = """from autoai_libs.cognito.transforms.transform_utils import TNoOp
from lightgbm import LGBMClassifier
from lale.operators import make_pipeline

t_no_op = TNoOp(
    fun="fun",
    name="no_action",
    datatypes="x",
    feat_constraints=[],
    tgraph="tgraph",
)
lgbm_classifier = LGBMClassifier(
    class_weight="balanced", learning_rate=0.18, n_estimators=100
)
pipeline = make_pipeline(t_no_op, lgbm_classifier)"""
        self._roundtrip(
            expected, lale.pretty_print.to_string(pipeline, combinators=False)
        )

    def test_autoai_libs_two_ops_with_combinator(self):
        from autoai_libs.transformers.exportable import (
            CompressStrings,
            NumpyColumnSelector,
        )

        numpy_column_selector = NumpyColumnSelector(columns=[0, 2, 3, 5])
        compress_strings = CompressStrings(
            compress_type="hash",
            dtypes_list=["char_str", "char_str", "char_str", "char_str"],
            misslist_list=[[], [], [], []],
        )
        pipeline = lale.operators.make_pipeline(numpy_column_selector, compress_strings)
        expected = """from autoai_libs.transformers.exportable import NumpyColumnSelector
from autoai_libs.transformers.exportable import CompressStrings
import lale

lale.wrap_imported_operators()
numpy_column_selector = NumpyColumnSelector(columns=[0, 2, 3, 5])
compress_strings = CompressStrings(
    compress_type="hash",
    dtypes_list=["char_str", "char_str", "char_str", "char_str"],
    missing_values_reference_list=["?", "", "-", float("nan")],
    misslist_list=[[], [], [], []],
)
pipeline = numpy_column_selector >> compress_strings"""
        printed = lale.pretty_print.to_string(pipeline, combinators=True)
        try:
            self._roundtrip(expected, printed)
        except BaseException:
            expected = """from autoai_libs.transformers.exportable import NumpyColumnSelector
from autoai_libs.transformers.exportable import CompressStrings
import lale

lale.wrap_imported_operators()
numpy_column_selector = NumpyColumnSelector(columns=[0, 2, 3, 5])
compress_strings = CompressStrings(
    compress_type="hash",
    dtypes_list=["char_str", "char_str", "char_str", "char_str"],
    missing_values_reference_list=("?", "", "-", float("nan")),
    misslist_list=[[], [], [], []],
)
pipeline = numpy_column_selector >> compress_strings"""
            self._roundtrip(expected, printed)

    def test_expression(self):
        from lale.expressions import it, mean
        from lale.lib.rasl import Aggregate, Join, Scan

        scan1 = Scan(table=it["table1.csv"])
        scan2 = Scan(table=it["table2.csv"])
        join = Join(pred=[it["table1.csv"].k1 == it["table2.csv"].k2])
        aggregate = Aggregate(columns={"talk_time|mean": mean(it.talk_time)})
        pipeline = (scan1 & scan2) >> join >> aggregate
        expected = """from lale.lib.rasl import Scan
from lale.expressions import it
from lale.lib.rasl import Join
from lale.lib.rasl import Aggregate
from lale.expressions import mean
import lale

lale.wrap_imported_operators()
scan_0 = Scan(table=it["table1.csv"])
scan_1 = Scan(table=it["table2.csv"])
join = Join(pred=[it["table1.csv"].k1 == it["table2.csv"].k2])
aggregate = Aggregate(columns={"talk_time|mean": mean(it.talk_time)})
pipeline = (scan_0 & scan_1) >> join >> aggregate"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_sklearn_pipeline(self):
        from lale.lib.sklearn import PCA, LogisticRegression, Pipeline

        pipeline = Pipeline(steps=[("pca", PCA), ("lr", LogisticRegression(C=0.1))])
        expected = """from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import lale

lale.wrap_imported_operators()
logistic_regression = LogisticRegression(C=0.1)
pipeline = Pipeline(steps=[("pca", PCA), ("lr", logistic_regression)])"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_sklearn_pipeline_2(self):
        from lale.lib.sklearn import PCA, LogisticRegression, Pipeline

        pipeline = Pipeline(steps=[("pca", PCA), ("lr", LogisticRegression(C=0.1))])
        expected = """from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression(C=0.1)
pipeline = Pipeline(steps=[("pca", PCA), ("lr", logistic_regression)])"""
        printed = lale.pretty_print.to_string(pipeline, astype="sklearn")
        self._roundtrip(expected, printed)

    def test_customize_schema_enum_and_number(self):
        from lale.lib.sklearn import LogisticRegression

        pipeline = LogisticRegression.customize_schema(
            solver={"enum": ["lbfgs", "liblinear"], "default": "liblinear"},
            tol={
                "type": "number",
                "minimum": 0.00001,
                "maximum": 0.1,
                "default": 0.0001,
            },
        )(solver="lbfgs")
        expected = """from sklearn.linear_model import LogisticRegression
import lale

lale.wrap_imported_operators()
pipeline = LogisticRegression.customize_schema(
    solver={"enum": ["lbfgs", "liblinear"], "default": "liblinear"},
    tol={
        "type": "number",
        "minimum": 1e-05,
        "maximum": 0.1,
        "default": 0.0001,
    },
)(solver="lbfgs")"""
        self._roundtrip(expected, pipeline.pretty_print(customize_schema=True))

    def test_customize_schema_none_and_boolean(self):
        from lale.lib.sklearn import RandomForestRegressor

        pipeline = RandomForestRegressor.customize_schema(
            bootstrap={"type": "boolean", "default": True},
            random_state={
                "anyOf": [
                    {"laleType": "numpy.random.RandomState"},
                    {
                        "description": "RandomState used by np.random",
                        "enum": [None],
                    },
                    {"description": "Explicit seed.", "type": "integer"},
                ],
                "default": 33,
            },
        )(n_estimators=50)
        expected = """from sklearn.ensemble import RandomForestRegressor
import lale

lale.wrap_imported_operators()
pipeline = RandomForestRegressor.customize_schema(
    bootstrap={"type": "boolean", "default": True},
    random_state={
        "anyOf": [
            {"laleType": "numpy.random.RandomState"},
            {"description": "RandomState used by np.random", "enum": [None]},
            {"description": "Explicit seed.", "type": "integer"},
        ],
        "default": 33,
    },
)(n_estimators=50)"""
        # this should not include "random_state=33" because that would be
        # redundant with the schema, and would prevent automated search
        self._roundtrip(expected, pipeline.pretty_print(customize_schema=True))

    def test_customize_schema_print_defaults(self):
        from lale.lib.sklearn import RandomForestRegressor

        pipeline = RandomForestRegressor.customize_schema(
            bootstrap={"type": "boolean", "default": True},  # default unchanged
            random_state={
                "anyOf": [
                    {"laleType": "numpy.random.RandomState"},
                    {"enum": [None]},
                    {"type": "integer"},
                ],
                "default": 33,  # default changed
            },
        )(n_estimators=50)
        expected = """from sklearn.ensemble import RandomForestRegressor
import lale

lale.wrap_imported_operators()
pipeline = RandomForestRegressor(n_estimators=50, random_state=33)"""
        # print exactly those defaults that changed
        self._roundtrip(expected, pipeline.pretty_print(customize_schema=False))

    def test_user_operator_in_toplevel_module(self):
        import importlib
        import os.path
        import sys
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as tmp_py_file:
            file_contents = """import numpy as np
import lale.operators

class _MockClassifierImpl:
    def __init__(self, int_hp=0):
        self.int_hp = int_hp

    def fit(self, X, y):
        self.some_y = list(y)[0]

    def predict(self, X):
        return self.some_y

MockClassifier = lale.operators.make_operator(_MockClassifierImpl)
"""
            tmp_py_file.write(file_contents)
            tmp_py_file.flush()
            dir_name = os.path.dirname(tmp_py_file.name)
            old_pythonpath = sys.path
            try:
                sys.path.append(dir_name)
                module_name = os.path.basename(tmp_py_file.name)[: -len(".py")]
                module = importlib.import_module(module_name)
                MockClf = getattr(module, "MockClassifier")
                self.assertIsInstance(MockClf, lale.operators.PlannedIndividualOp)
                self.assertEqual(MockClf.name(), "MockClassifier")
                pipeline = MockClf(int_hp=42)
                expected = f"""from {module_name} import MockClassifier as MockClf
import lale

lale.wrap_imported_operators(["{module_name}"])
pipeline = MockClf(int_hp=42)"""
                self._roundtrip(expected, pipeline.pretty_print())
            finally:
                sys.path = old_pythonpath

    def test_nonlib_operator(self):
        from test.mock_custom_operators import CustomOrigOperator

        from lale.lib.sklearn import LogisticRegression

        pipeline = CustomOrigOperator() >> LogisticRegression()
        expected = """from test.mock_module import CustomOrigOperator
from sklearn.linear_model import LogisticRegression
import lale

lale.wrap_imported_operators(["test.mock_module"])
pipeline = CustomOrigOperator() >> LogisticRegression()"""
        self._roundtrip(expected, pipeline.pretty_print())

    @unittest.skip("TODO: avoid spurious 'name' keys in printed dictionaries")
    def test_fairness_info(self):
        from lale.lib.aif360 import DisparateImpactRemover, fetch_creditg_df
        from lale.lib.lale import Hyperopt, Project
        from lale.lib.sklearn import KNeighborsClassifier

        X, y, fairness_info = fetch_creditg_df()
        disparate_impact_remover = DisparateImpactRemover(
            **fairness_info,
            preparation=Project(columns={"type": "number"}),
        )
        planned = disparate_impact_remover >> KNeighborsClassifier()
        frozen = planned.freeze_trainable()
        pipeline = frozen.auto_configure(X, y, optimizer=Hyperopt, cv=2, max_evals=1)
        expected = """from aif360.algorithms.preprocessing import DisparateImpactRemover
from lale.lib.rasl import Project
from sklearn.neighbors import KNeighborsClassifier
import lale

lale.wrap_imported_operators()
project = Project(columns={"type": "number"})
disparate_impact_remover = DisparateImpactRemover(
    favorable_labels=["good"],
    protected_attributes=[
        {
            "reference_group": [
                "male div/sep", "male mar/wid", "male single",
            ],
            "feature": "personal_status",
        },
        {
            "reference_group": [[26, 1000]],
            "feature": "age",
        },
    ],
    preparation=project,
)
pipeline = disparate_impact_remover >> KNeighborsClassifier()"""
        self._roundtrip(expected, pipeline.pretty_print())


class TestToAndFromJSON(unittest.TestCase):
    def test_trainable_individual_op(self):
        self.maxDiff = None
        from lale.json_operator import from_json, to_json
        from lale.lib.sklearn import LogisticRegression as LR

        operator = LR(LR.enum.solver.sag, C=0.1)
        json_expected = {
            "class": LR.class_name(),
            "state": "trainable",
            "operator": "LogisticRegression",
            "label": "LR",
            "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html",
            "hyperparams": {"C": 0.1, "solver": "sag"},
            "is_frozen_trainable": False,
        }
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json_2, json_expected)

    def test_operator_choice(self):
        self.maxDiff = None
        from lale.json_operator import from_json, to_json
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import MinMaxScaler as Scl

        operator = PCA | Scl
        json_expected = {
            "class": "lale.operators.OperatorChoice",
            "operator": "OperatorChoice",
            "state": "planned",
            "steps": {
                "pca": {
                    "class": PCA.class_name(),
                    "state": "planned",
                    "operator": "PCA",
                    "label": "PCA",
                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html",
                },
                "scl": {
                    "class": Scl.class_name(),
                    "state": "planned",
                    "operator": "MinMaxScaler",
                    "label": "Scl",
                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.min_max_scaler.html",
                },
            },
        }
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json_2, json_expected)

    def test_pipeline_1(self):
        self.maxDiff = None
        from lale.json_operator import from_json, to_json
        from lale.lib.lale import NoOp
        from lale.lib.rasl import ConcatFeatures
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import LogisticRegression as LR

        operator = (PCA & NoOp) >> ConcatFeatures >> LR
        json_expected = {
            "class": "lale.operators.PlannedPipeline",
            "state": "planned",
            "edges": [
                ["pca", "concat_features"],
                ["no_op", "concat_features"],
                ["concat_features", "lr"],
            ],
            "steps": {
                "pca": {
                    "class": PCA.class_name(),
                    "state": "planned",
                    "operator": "PCA",
                    "label": "PCA",
                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html",
                },
                "no_op": {
                    "class": NoOp.class_name(),
                    "state": "trained",
                    "operator": "NoOp",
                    "label": "NoOp",
                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.no_op.html",
                    "hyperparams": None,
                    "coefs": None,
                    "is_frozen_trainable": True,
                    "is_frozen_trained": True,
                },
                "concat_features": {
                    "class": ConcatFeatures.class_name(),
                    "state": "trained",
                    "operator": "ConcatFeatures",
                    "label": "ConcatFeatures",
                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.rasl.concat_features.html",
                    "hyperparams": None,
                    "coefs": None,
                    "is_frozen_trainable": True,
                    "is_frozen_trained": True,
                },
                "lr": {
                    "class": LR.class_name(),
                    "state": "planned",
                    "operator": "LogisticRegression",
                    "label": "LR",
                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html",
                },
            },
        }
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json, json_2)

    def test_pipeline_2(self):
        from lale.json_operator import from_json, to_json
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import (
            PCA,
            KNeighborsClassifier,
            LogisticRegression,
            Nystroem,
        )
        from lale.operators import make_choice, make_pipeline

        kernel_tfm_or_not = make_choice(NoOp, Nystroem)
        tfm = PCA
        clf = make_choice(LogisticRegression, KNeighborsClassifier)
        operator = make_pipeline(kernel_tfm_or_not, tfm, clf)
        json = to_json(operator)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json, json_2)

    def test_higher_order_1(self):
        from lale.json_operator import from_json
        from lale.lib.lale import Both
        from lale.lib.sklearn import PCA, Nystroem

        operator = Both(op1=PCA(n_components=2), op2=Nystroem)
        json_expected = {
            "class": Both.class_name(),
            "state": "trainable",
            "operator": "Both",
            "label": "Both",
            "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.both.html",
            "hyperparams": {
                "op1": {"$ref": "../steps/pca"},
                "op2": {"$ref": "../steps/nystroem"},
            },
            "steps": {
                "pca": {
                    "class": PCA.class_name(),
                    "state": "trainable",
                    "operator": "PCA",
                    "label": "PCA",
                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html",
                    "hyperparams": {"n_components": 2},
                    "is_frozen_trainable": False,
                },
                "nystroem": {
                    "class": Nystroem.class_name(),
                    "state": "planned",
                    "operator": "Nystroem",
                    "label": "Nystroem",
                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.nystroem.html",
                },
            },
            "is_frozen_trainable": False,
        }
        json = operator.to_json()
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = operator_2.to_json()
        self.assertEqual(json, json_2)

    def test_higher_order_2(self):
        self.maxDiff = None
        from lale.json_operator import from_json
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.lib.sklearn import VotingClassifier as Vote

        operator = Vote(
            estimators=[("knn", KNN), ("pipeline", PCA() >> LR)], voting="soft"
        )
        json_expected = {
            "class": Vote.class_name(),
            "state": "trainable",
            "operator": "VotingClassifier",
            "is_frozen_trainable": True,
            "label": "Vote",
            "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.voting_classifier.html",
            "hyperparams": {
                "estimators": [
                    ("knn", {"$ref": "../steps/knn"}),
                    ("pipeline", {"$ref": "../steps/pipeline"}),
                ],
                "voting": "soft",
            },
            "steps": {
                "knn": {
                    "class": KNN.class_name(),
                    "state": "planned",
                    "operator": "KNeighborsClassifier",
                    "label": "KNN",
                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.k_neighbors_classifier.html",
                },
                "pipeline": {
                    "class": "lale.operators.PlannedPipeline",
                    "state": "planned",
                    "edges": [["pca", "lr"]],
                    "steps": {
                        "pca": {
                            "class": PCA.class_name(),
                            "state": "trainable",
                            "operator": "PCA",
                            "label": "PCA",
                            "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html",
                            "hyperparams": {},
                            "is_frozen_trainable": False,
                        },
                        "lr": {
                            "class": LR.class_name(),
                            "state": "planned",
                            "operator": "LogisticRegression",
                            "label": "LR",
                            "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html",
                        },
                    },
                },
            },
        }
        json = operator.to_json()
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = operator_2.to_json()
        self.assertEqual(json, json_2)

    def test_nested(self):
        self.maxDiff = None
        from lale.json_operator import from_json, to_json
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import LogisticRegression as LR

        operator = PCA >> (LR(C=0.09) | NoOp >> LR(C=0.19))
        json_expected = {
            "class": "lale.operators.PlannedPipeline",
            "state": "planned",
            "edges": [["pca", "choice"]],
            "steps": {
                "pca": {
                    "class": PCA.class_name(),
                    "state": "planned",
                    "operator": "PCA",
                    "label": "PCA",
                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html",
                },
                "choice": {
                    "class": "lale.operators.OperatorChoice",
                    "state": "planned",
                    "operator": "OperatorChoice",
                    "steps": {
                        "lr_0": {
                            "class": LR.class_name(),
                            "state": "trainable",
                            "operator": "LogisticRegression",
                            "label": "LR",
                            "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html",
                            "hyperparams": {"C": 0.09},
                            "is_frozen_trainable": False,
                        },
                        "pipeline_1": {
                            "class": "lale.operators.TrainablePipeline",
                            "state": "trainable",
                            "edges": [["no_op", "lr_1"]],
                            "steps": {
                                "no_op": {
                                    "class": NoOp.class_name(),
                                    "state": "trained",
                                    "operator": "NoOp",
                                    "label": "NoOp",
                                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.no_op.html",
                                    "hyperparams": None,
                                    "coefs": None,
                                    "is_frozen_trainable": True,
                                    "is_frozen_trained": True,
                                },
                                "lr_1": {
                                    "class": LR.class_name(),
                                    "state": "trainable",
                                    "operator": "LogisticRegression",
                                    "label": "LR",
                                    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html",
                                    "hyperparams": {"C": 0.19},
                                    "is_frozen_trainable": False,
                                },
                            },
                        },
                    },
                },
            },
        }
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json, json_2)

    def test_customize_schema(self):
        from lale.json_operator import from_json, to_json
        from lale.lib.sklearn import LogisticRegression as LR

        operator = LR.customize_schema(
            solver={"enum": ["lbfgs", "liblinear"], "default": "liblinear"},
            tol={
                "type": "number",
                "minimum": 0.00001,
                "maximum": 0.1,
                "default": 0.0001,
            },
        )
        json_expected = {
            "class": LR.class_name(),
            "state": "planned",
            "operator": "LogisticRegression",
            "label": "LR",
            "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html",
            "customize_schema": {
                "properties": {
                    "hyperparams": {
                        "allOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "solver": {
                                        "default": "liblinear",
                                        "enum": ["lbfgs", "liblinear"],
                                    },
                                    "tol": {
                                        "type": "number",
                                        "minimum": 0.00001,
                                        "maximum": 0.1,
                                        "default": 0.0001,
                                    },
                                },
                            }
                        ]
                    }
                }
            },
        }
        json = to_json(operator)
        self.maxDiff = None
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json, json_2)


class TestDiff(unittest.TestCase):
    def test_single_op(self):
        from lale.lib.sklearn import LogisticRegression

        single_op = LogisticRegression()
        single_op_param = LogisticRegression(solver="saga")

        expected_diff = (
            "  from sklearn.linear_model import LogisticRegression\n"
            "  import lale\n"
            "  \n"
            "  lale.wrap_imported_operators()\n"
            "- pipeline = LogisticRegression()\n"
            '+ pipeline = LogisticRegression(solver="saga")\n'
            "?                               +++++++++++++\n"
        )
        diff_str = single_op.diff(single_op_param, ipython_display=False)
        self.assertEqual(diff_str, expected_diff)

        expected_diff_reverse = (
            "  from sklearn.linear_model import LogisticRegression\n"
            "  import lale\n"
            "  \n"
            "  lale.wrap_imported_operators()\n"
            '- pipeline = LogisticRegression(solver="saga")\n'
            "?                               -------------\n\n"
            "+ pipeline = LogisticRegression()"
        )
        diff_str_reverse = single_op_param.diff(single_op, ipython_display=False)
        self.assertEqual(diff_str_reverse, expected_diff_reverse)

    def test_pipeline(self):
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import PCA, LogisticRegression, SelectKBest

        pipeline_simple = PCA >> SelectKBest >> LogisticRegression
        pipeline_choice = (PCA | NoOp) >> SelectKBest >> LogisticRegression

        expected_diff = (
            "  from sklearn.decomposition import PCA\n"
            "+ from lale.lib.lale import NoOp\n"
            "  from sklearn.feature_selection import SelectKBest\n"
            "  from sklearn.linear_model import LogisticRegression\n"
            "  import lale\n"
            "  \n"
            "  lale.wrap_imported_operators()\n"
            "- pipeline = PCA >> SelectKBest >> LogisticRegression\n"
            "+ pipeline = (PCA | NoOp) >> SelectKBest >> LogisticRegression\n"
            "?            +   ++++++++\n"
        )
        diff_str = pipeline_simple.diff(pipeline_choice, ipython_display=False)
        self.assertEqual(diff_str, expected_diff)

        expected_diff_reverse = (
            "  from sklearn.decomposition import PCA\n"
            "- from lale.lib.lale import NoOp\n"
            "  from sklearn.feature_selection import SelectKBest\n"
            "  from sklearn.linear_model import LogisticRegression\n"
            "  import lale\n"
            "  \n"
            "  lale.wrap_imported_operators()\n"
            "- pipeline = (PCA | NoOp) >> SelectKBest >> LogisticRegression\n"
            "?            -   --------\n\n"
            "+ pipeline = PCA >> SelectKBest >> LogisticRegression"
        )
        diff_str_reverse = pipeline_choice.diff(pipeline_simple, ipython_display=False)
        self.assertEqual(diff_str_reverse, expected_diff_reverse)

    def test_single_op_pipeline(self):
        from lale.lib.sklearn import PCA, LogisticRegression, SelectKBest

        single_op = LogisticRegression()
        pipeline = PCA >> SelectKBest >> LogisticRegression

        expected_diff = (
            "+ from sklearn.decomposition import PCA\n"
            "+ from sklearn.feature_selection import SelectKBest\n"
            "  from sklearn.linear_model import LogisticRegression\n"
            "  import lale\n"
            "  \n"
            "  lale.wrap_imported_operators()\n"
            "- pipeline = LogisticRegression()\n"
            "+ pipeline = PCA >> SelectKBest >> LogisticRegression"
        )
        diff_str = single_op.diff(pipeline, ipython_display=False)
        self.assertEqual(expected_diff, diff_str)

        expected_diff_reverse = (
            "- from sklearn.decomposition import PCA\n"
            "- from sklearn.feature_selection import SelectKBest\n"
            "  from sklearn.linear_model import LogisticRegression\n"
            "  import lale\n"
            "  \n"
            "  lale.wrap_imported_operators()\n"
            "- pipeline = PCA >> SelectKBest >> LogisticRegression\n"
            "+ pipeline = LogisticRegression()"
        )
        diff_str_reverse = pipeline.diff(single_op, ipython_display=False)
        self.assertEqual(expected_diff_reverse, diff_str_reverse)

    def test_options(self):
        from lale.lib.sklearn import LogisticRegression

        single_op = LogisticRegression()
        single_op_schema = single_op.customize_schema(solver={"enum": ["saga"]})

        expected_diff_no_imports = "  pipeline = LogisticRegression()"
        diff_str_no_imports = single_op.diff(
            single_op_schema, show_imports=False, ipython_display=False
        )
        self.assertEqual(diff_str_no_imports, expected_diff_no_imports)

        expected_diff_no_schema = (
            "  from sklearn.linear_model import LogisticRegression\n"
            "  import lale\n"
            "  \n"
            "  lale.wrap_imported_operators()\n"
            "- pipeline = LogisticRegression()\n"
            '+ pipeline = LogisticRegression.customize_schema(solver={"enum": ["saga"]})()'
        )
        diff_str_no_schema = single_op.diff(
            single_op_schema, customize_schema=True, ipython_display=False
        )
        self.assertEqual(diff_str_no_schema, expected_diff_no_schema)
