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
import lale.pretty_print

class TestToGraphviz(unittest.TestCase):
    def test_with_operator_choice(self):
        from lale.operators import make_choice
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import KNeighborsClassifier
        from lale.lib.sklearn import LogisticRegression
        from lale.lib.sklearn import Nystroem
        from lale.lib.sklearn import PCA
        kernel_tfm_or_not =  NoOp | Nystroem
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
        self.assertEqual(expected, printed)
        globals2 = {}
        exec(printed, globals2)
        pipeline2 = globals2['pipeline']

    def test_indiv_op_1(self):
        from lale.lib.sklearn import LogisticRegression
        pipeline = LogisticRegression(solver='saga', C=0.9)
        expected = """from lale.lib.sklearn import LogisticRegression
import lale
lale.wrap_imported_operators()

pipeline = LogisticRegression(solver='saga', C=0.9)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_indiv_op_2(self):
        from lale.lib.sklearn import LogisticRegression
        pipeline = LogisticRegression()
        expected = """from lale.lib.sklearn import LogisticRegression
import lale
lale.wrap_imported_operators()

pipeline = LogisticRegression()"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_reducible(self):
        from lale.lib.sklearn import MinMaxScaler
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import Nystroem
        from lale.lib.lale import ConcatFeatures
        from lale.lib.sklearn import KNeighborsClassifier
        from lale.lib.sklearn import LogisticRegression
        pca = PCA(copy=False)
        logistic_regression = LogisticRegression(solver='saga', C=0.9)
        pipeline = (MinMaxScaler | NoOp) >> (pca & Nystroem) >> ConcatFeatures >> (KNeighborsClassifier | logistic_regression)
        expected = \
"""from lale.lib.sklearn import MinMaxScaler
from lale.lib.lale import NoOp
from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
from lale.lib.lale import ConcatFeatures
from lale.lib.sklearn import KNeighborsClassifier
from lale.lib.sklearn import LogisticRegression
import lale
lale.wrap_imported_operators()

pca = PCA(copy=False)
logistic_regression = LogisticRegression(solver='saga', C=0.9)
pipeline = (MinMaxScaler | NoOp) >> (pca & Nystroem) >> ConcatFeatures >> (KNeighborsClassifier | logistic_regression)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_no_combinators(self):
        from lale.lib.sklearn import MinMaxScaler
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import Nystroem
        from lale.lib.lale import ConcatFeatures
        from lale.lib.sklearn import KNeighborsClassifier
        from lale.lib.sklearn import LogisticRegression
        pca = PCA(copy=False)
        logistic_regression = LogisticRegression(solver='saga', C=0.9)
        pipeline = (MinMaxScaler | NoOp) >> (pca & Nystroem & NoOp) >> ConcatFeatures >> (KNeighborsClassifier | logistic_regression)
        expected = \
"""from lale.lib.sklearn import MinMaxScaler
from lale.lib.lale import NoOp
from lale.operators import make_choice
from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
from lale.operators import make_union
from lale.lib.sklearn import KNeighborsClassifier
from lale.lib.sklearn import LogisticRegression
from lale.operators import make_pipeline

choice_0 = make_choice(MinMaxScaler, NoOp)
pca = PCA(copy=False)
union = make_union(pca, Nystroem, NoOp)
logistic_regression = LogisticRegression(solver='saga', C=0.9)
choice_1 = make_choice(KNeighborsClassifier, logistic_regression)
pipeline = make_pipeline(choice_0, union, choice_1)"""
        printed = lale.pretty_print.to_string(pipeline, combinators=False)
        self._roundtrip(expected, printed)

    def test_import_as_1(self):
        from lale.lib.sklearn import LogisticRegression as LR
        pipeline = LR(solver='saga', C=0.9)
        expected = """from lale.lib.sklearn import LogisticRegression as LR
import lale
lale.wrap_imported_operators()

pipeline = LR(solver='saga', C=0.9)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_import_as_2(self):
        from lale.lib.sklearn import MinMaxScaler as Scaler
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import Nystroem
        from lale.lib.lale import ConcatFeatures as Concat
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import LogisticRegression as LR
        pca = PCA(copy=False)
        lr = LR(solver='saga', C=0.9)
        pipeline = (Scaler | NoOp) >> (pca & Nystroem) >> Concat >> (KNN | lr)
        expected = \
"""from lale.lib.sklearn import MinMaxScaler as Scaler
from lale.lib.lale import NoOp
from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
from lale.lib.lale import ConcatFeatures as Concat
from lale.lib.sklearn import KNeighborsClassifier as KNN
from lale.lib.sklearn import LogisticRegression as LR
import lale
lale.wrap_imported_operators()

pca = PCA(copy=False)
lr = LR(solver='saga', C=0.9)
pipeline = (Scaler | NoOp) >> (pca & Nystroem) >> Concat >> (KNN | lr)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_operator_choice(self):
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import MinMaxScaler as Scl
        pipeline = PCA | Scl
        expected = \
"""from lale.lib.sklearn import PCA
from lale.lib.sklearn import MinMaxScaler as Scl
import lale
lale.wrap_imported_operators()

pipeline = PCA | Scl"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_higher_order(self):
        from lale.lib.lale import Both
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import Nystroem
        pipeline = Both(op1=PCA(n_components=2), op2=Nystroem)
        expected = """from lale.lib.lale import Both
from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
import lale
lale.wrap_imported_operators()

pca = PCA(n_components=2)
pipeline = Both(op1=pca, op2=Nystroem)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_higher_order_2(self):
        from lale.lib.sklearn import VotingClassifier as Vote
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import LogisticRegression as LR
        pipeline = Vote(estimators=[('knn',KNN), ('pipeline',PCA()>>LR)],
                        voting='soft')
        expected = """from lale.lib.sklearn import VotingClassifier as Vote
from lale.lib.sklearn import KNeighborsClassifier as KNN
from lale.lib.sklearn import PCA
from lale.lib.sklearn import LogisticRegression as LR
import lale
lale.wrap_imported_operators()

pipeline = Vote(estimators=[('knn', KNN), ('pipeline', PCA() >> LR)], voting='soft')"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_multimodal(self):
        from lale.lib.lale import Project
        from lale.lib.sklearn import Normalizer as Norm
        from lale.lib.sklearn import OneHotEncoder as OneHot
        from lale.lib.lale import ConcatFeatures as Cat
        from lale.lib.sklearn import LinearSVC
        project_0 = Project(columns={'type': 'number'})
        project_1 = Project(columns={'type': 'string'})
        linear_svc = LinearSVC(C=29617.4, dual=False, tol=0.005266)
        pipeline = ((project_0 >> Norm()) & (project_1 >> OneHot())) >> Cat >> linear_svc
        expected = \
"""from lale.lib.lale import Project
from lale.lib.sklearn import Normalizer as Norm
from lale.lib.sklearn import OneHotEncoder as OneHot
from lale.lib.lale import ConcatFeatures as Cat
from lale.lib.sklearn import LinearSVC
import lale
lale.wrap_imported_operators()

project_0 = Project(columns={'type': 'number'})
project_1 = Project(columns={'type': 'string'})
linear_svc = LinearSVC(C=29617.4, dual=False, tol=0.005266)
pipeline = ((project_0 >> Norm()) & (project_1 >> OneHot())) >> Cat >> linear_svc"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_irreducible_1(self):
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import Nystroem
        from lale.lib.sklearn import MinMaxScaler
        from lale.lib.sklearn import LogisticRegression
        from lale.lib.sklearn import KNeighborsClassifier
        from lale.operators import get_pipeline_of_applicable_type
        choice = PCA | Nystroem
        pipeline = get_pipeline_of_applicable_type(
            steps=[choice, MinMaxScaler, LogisticRegression, KNeighborsClassifier],
            edges=[(choice,LogisticRegression), (MinMaxScaler,LogisticRegression), (MinMaxScaler,KNeighborsClassifier)])
        expected = \
"""from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
from lale.lib.sklearn import MinMaxScaler
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import KNeighborsClassifier
from lale.operators import get_pipeline_of_applicable_type
import lale
lale.wrap_imported_operators()

choice = PCA | Nystroem
pipeline = get_pipeline_of_applicable_type(steps=[choice, MinMaxScaler, LogisticRegression, KNeighborsClassifier], edges=[(choice,LogisticRegression), (MinMaxScaler,LogisticRegression), (MinMaxScaler,KNeighborsClassifier)])"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_irreducible_2(self):
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import MinMaxScaler as MMS
        from lale.lib.lale import ConcatFeatures as HStack
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.operators import get_pipeline_of_applicable_type
        pipeline_0 = HStack >> LR
        pipeline = get_pipeline_of_applicable_type(
            steps=[PCA, MMS, KNN, pipeline_0],
            edges=[(PCA, KNN), (PCA, pipeline_0), (MMS, pipeline_0)])
        expected = \
"""from lale.lib.sklearn import PCA
from lale.lib.sklearn import MinMaxScaler as MMS
from lale.lib.sklearn import KNeighborsClassifier as KNN
from lale.lib.lale import ConcatFeatures as HStack
from lale.lib.sklearn import LogisticRegression as LR
from lale.operators import get_pipeline_of_applicable_type
import lale
lale.wrap_imported_operators()

pipeline_0 = HStack >> LR
pipeline = get_pipeline_of_applicable_type(steps=[PCA, MMS, KNN, pipeline_0], edges=[(PCA,KNN), (PCA,pipeline_0), (MMS,pipeline_0)])"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_nested(self):
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.lib.lale import NoOp
        lr_0 = LR(C=0.09)
        lr_1 = LR(C=0.19)
        pipeline = PCA >> (lr_0 | NoOp >> lr_1)
        expected = \
"""from lale.lib.sklearn import PCA
from lale.lib.sklearn import LogisticRegression as LR
from lale.lib.lale import NoOp
import lale
lale.wrap_imported_operators()

lr_0 = LR(C=0.09)
lr_1 = LR(C=0.19)
pipeline = PCA >> (lr_0 | NoOp >> lr_1)"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_cat_encoder(self):
        from autoai_libs.transformers.exportable import CatEncoder
        import numpy as np
        from lale.lib.sklearn import LogisticRegression as LR
        cat_encoder = CatEncoder(categories='auto', dtype=np.float64, encoding='ordinal', handle_unknown='error')
        pipeline = cat_encoder >> LR()
        expected = \
"""from autoai_libs.transformers.exportable import CatEncoder
import numpy as np
from lale.lib.sklearn import LogisticRegression as LR
import lale
lale.wrap_imported_operators()

cat_encoder = CatEncoder(categories='auto', dtype=np.float64, encoding='ordinal', handle_unknown='error')
pipeline = cat_encoder >> LR()"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_numpy_replace_missing_values(self):
        from autoai_libs.transformers.exportable import NumpyReplaceMissingValues
        from lale.lib.sklearn import LogisticRegression as LR
        numpy_replace_missing_values = NumpyReplaceMissingValues(
            filling_values=float('nan'), missing_values=['?'])
        pipeline = numpy_replace_missing_values >> LR()
        expected = \
"""from autoai_libs.transformers.exportable import NumpyReplaceMissingValues
from lale.lib.sklearn import LogisticRegression as LR
import lale
lale.wrap_imported_operators()

numpy_replace_missing_values = NumpyReplaceMissingValues(filling_values=float('nan'), missing_values=['?'])
pipeline = numpy_replace_missing_values >> LR()"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))

    def test_autoai_libs_tam(self):
        from autoai_libs.cognito.transforms.transform_utils import TAM
        import autoai_libs.cognito.transforms.transform_extras
        import numpy as np
        from lale.lib.sklearn import LogisticRegression as LR
        tam = TAM(tans_class=autoai_libs.cognito.transforms.transform_extras.IsolationForestAnomaly, name='isoforestanomaly', col_names=['a', 'b', 'c'], col_dtypes=[np.dtype('float32'), np.dtype('float32'), np.dtype('float32')])
        pipeline = tam >> LR()
        expected = \
"""from autoai_libs.cognito.transforms.transform_utils import TAM
import autoai_libs.cognito.transforms.transform_extras
import numpy as np
from lale.lib.sklearn import LogisticRegression as LR
import lale
lale.wrap_imported_operators()

tam = TAM(tans_class=autoai_libs.cognito.transforms.transform_extras.IsolationForestAnomaly, name='isoforestanomaly', col_names=['a', 'b', 'c'], col_dtypes=[np.dtype('float32'), np.dtype('float32'), np.dtype('float32')])
pipeline = tam >> LR()"""
        self._roundtrip(expected, lale.pretty_print.to_string(pipeline))


class TestToAndFromJSON(unittest.TestCase):
    def test_trainable_individual_op(self):
        self.maxDiff = None
        from lale.json_operator import to_json, from_json
        from lale.lib.sklearn import LogisticRegression as LR
        operator = LR(LR.solver.sag, C=0.1)
        json_expected = {
            'class': 'lale.lib.sklearn.logistic_regression.LogisticRegressionImpl',
            'state': 'trainable',
            'operator': 'LogisticRegression', 'label': 'LR',
            'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html',
            'hyperparams': {'C': 0.1, 'solver': 'sag'},
            'is_frozen_trainable': False}
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json_2, json_expected)

    def test_operator_choice(self):
        self.maxDiff = None
        from lale.json_operator import to_json, from_json
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import MinMaxScaler as Scl
        operator = PCA | Scl
        json_expected = {
          'class': 'lale.operators.OperatorChoice',
          'operator': 'OperatorChoice',
          'state': 'planned',
          'steps': {
            'pca': {
              'class': 'lale.lib.sklearn.pca.PCAImpl',
              'state': 'planned',
              'operator': 'PCA', 'label': 'PCA',
              'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html'},
            'scl': {
              'class': 'lale.lib.sklearn.min_max_scaler.MinMaxScalerImpl',
              'state': 'planned',
              'operator': 'MinMaxScaler', 'label': 'Scl',
              'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html'}}}
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json_2, json_expected)

    def test_pipeline_1(self):
        self.maxDiff = None
        from lale.json_operator import to_json, from_json
        from lale.lib.lale import ConcatFeatures, NoOp
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.lib.sklearn import PCA
        operator = (PCA & NoOp) >> ConcatFeatures >> LR
        json_expected = {
          'class': 'lale.operators.PlannedPipeline',
          'state': 'planned',
          'edges': [
              ['pca', 'concat_features'],
              ['no_op', 'concat_features'],
              ['concat_features', 'lr']],
          'steps': {
            'pca': {
              'class': 'lale.lib.sklearn.pca.PCAImpl',
              'state': 'planned',
              'operator': 'PCA', 'label': 'PCA',
              'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html'},
            'no_op': {
              'class': 'lale.lib.lale.no_op.NoOpImpl',
              'state': 'trained',
              'operator': 'NoOp', 'label': 'NoOp',
              'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.no_op.html',
              'hyperparams': None,
              'coefs': None,
              'is_frozen_trainable': True, 'is_frozen_trained': True},
            'concat_features': {
              'class': 'lale.lib.lale.concat_features.ConcatFeaturesImpl',
              'state': 'trained',
              'operator': 'ConcatFeatures', 'label': 'ConcatFeatures',
              'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.concat_features.html',
              'hyperparams': None,
              'coefs': None,
              'is_frozen_trainable': True, 'is_frozen_trained': True},
            'lr': {
              'class': 'lale.lib.sklearn.logistic_regression.LogisticRegressionImpl',
              'state': 'planned',
              'operator': 'LogisticRegression', 'label': 'LR',
              'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html'}}}
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json, json_2)

    def test_pipeline_2(self):
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import Nystroem
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import LogisticRegression
        from lale.lib.sklearn import KNeighborsClassifier
        from lale.operators import make_choice, make_pipeline
        from lale.json_operator import to_json, from_json
        kernel_tfm_or_not =  make_choice(NoOp, Nystroem)
        tfm = PCA
        clf = make_choice(LogisticRegression, KNeighborsClassifier)
        operator = make_pipeline(kernel_tfm_or_not, tfm, clf)
        json = to_json(operator)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json, json_2)

    def test_higher_order_1(self):
        from lale.lib.lale import Both
        from lale.lib.sklearn import PCA, Nystroem
        from lale.json_operator import from_json
        operator = Both(op1=PCA(n_components=2), op2=Nystroem)
        json_expected = {
          'class': 'lale.lib.lale.both.BothImpl',
          'state': 'trainable',
          'operator': 'Both', 'label': 'Both',
          'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.both.html',
          'hyperparams': {
            'op1': {'$ref': '../steps/pca'},
            'op2': {'$ref': '../steps/nystroem'}},
          'steps': {
            'pca': {
              'class': 'lale.lib.sklearn.pca.PCAImpl',
              'state': 'trainable',
              'operator': 'PCA', 'label': 'PCA',
              'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html',
              'hyperparams': {'n_components': 2},
              'is_frozen_trainable': False},
            'nystroem': {
              'class': 'lale.lib.sklearn.nystroem.NystroemImpl',
              'state': 'planned',
              'operator': 'Nystroem', 'label': 'Nystroem',
              'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html'}},
          'is_frozen_trainable': False}
        json = operator.to_json()
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = operator_2.to_json()
        self.assertEqual(json, json_2)

    def test_higher_order_2(self):
        self.maxDiff = None
        from lale.lib.sklearn import VotingClassifier as Vote
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.json_operator import from_json
        operator = Vote(estimators=[('knn',KNN), ('pipeline',PCA()>>LR)],
                        voting='soft')
        json_expected = {
          'class': 'lale.lib.sklearn.voting_classifier.VotingClassifierImpl',
          'state': 'trainable',
          'operator': 'VotingClassifier',
          'is_frozen_trainable': True,
          'label': 'Vote',
          'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html',
          'hyperparams': {
            'estimators': [
              ('knn', {'$ref': '../steps/knn'}),
              ('pipeline', {'$ref': '../steps/pipeline'})],
            'voting': 'soft'},
          'steps': {
            'knn': {
              'class': 'lale.lib.sklearn.k_neighbors_classifier.KNeighborsClassifierImpl',
              'state': 'planned',
              'operator': 'KNeighborsClassifier', 'label': 'KNN',
              'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'},
            'pipeline': {
              'class': 'lale.operators.PlannedPipeline',
              'state': 'planned',
              'edges': [['pca', 'lr']],
              'steps': {
                'pca': {
                  'class': 'lale.lib.sklearn.pca.PCAImpl',
                  'state': 'trainable',
                  'operator': 'PCA', 'label': 'PCA',
                  'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html',
                  'hyperparams': {},
                  'is_frozen_trainable': False},
                'lr': {
                  'class': 'lale.lib.sklearn.logistic_regression.LogisticRegressionImpl',
                  'state': 'planned',
                  'operator': 'LogisticRegression', 'label': 'LR',
                  'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html'}}}}}
        json = operator.to_json()
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = operator_2.to_json()
        self.assertEqual(json, json_2)

    def test_nested(self):
        self.maxDiff = None
        from lale.json_operator import to_json, from_json
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.lib.sklearn import PCA
        operator = PCA >> (LR(C=0.09) | NoOp >> LR(C=0.19))
        json_expected = {
          'class': 'lale.operators.PlannedPipeline',
          'state': 'planned',
          'edges': [['pca', 'choice']],
          'steps': {
            'pca': {
              'class': 'lale.lib.sklearn.pca.PCAImpl',
              'state': 'planned',
              'operator': 'PCA', 'label': 'PCA',
              'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.pca.html'},
            'choice': {
              'class': 'lale.operators.OperatorChoice',
              'state': 'planned',
              'operator': 'OperatorChoice',
              'steps': {
                'lr_0': {
                  'class': 'lale.lib.sklearn.logistic_regression.LogisticRegressionImpl',
                  'state': 'trainable',
                  'operator': 'LogisticRegression', 'label': 'LR',
                  'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html',
                  'hyperparams': {'C': 0.09},
                  'is_frozen_trainable': False},
                'pipeline_1': {
                  'class': 'lale.operators.TrainablePipeline',
                  'state': 'trainable',
                  'edges': [['no_op', 'lr_1']],
                  'steps': {
                    'no_op': {
                      'class': 'lale.lib.lale.no_op.NoOpImpl',
                      'state': 'trained',
                      'operator': 'NoOp', 'label': 'NoOp',
                      'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.no_op.html',
                      'hyperparams': None,
                      'coefs': None,
                      'is_frozen_trainable': True, 'is_frozen_trained': True},
                    'lr_1': {
                      'class': 'lale.lib.sklearn.logistic_regression.LogisticRegressionImpl',
                      'state': 'trainable',
                      'operator': 'LogisticRegression', 'label': 'LR',
                      'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html',
                      'hyperparams': {'C': 0.19},
                      'is_frozen_trainable': False}}}}}}}
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json, json_2)

if __name__ == '__main__':
    unittest.main()
