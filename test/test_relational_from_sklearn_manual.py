# Copyright 2021-2022 IBM Corporation
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

from lale.helpers import _ensure_pandas

from lale.lib.rasl import Convert
from lale.lib.rasl import MinMaxScaler as RaslMinMaxScaler
from lale.lib.rasl import SelectKBest as RaslSelectKBest
from lale.lib.rasl import OrdinalEncoder as RaslOrdinalEncoder
from test.test_relational_sklearn import _check_trained_min_max_scaler, _check_trained_ordinal_encoder

import sklearn
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest as SkSelectKBest
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

assert sklearn.__version__ >= "1.0", sklearn.__version__

def _check_data(self, sk_data, rasl_data, msg):
    rasl_data = _ensure_pandas(rasl_data)
    self.assertEqual(sk_data.shape, rasl_data.shape, msg)
    for row_idx in range(sk_data.shape[0]):
        for col_idx in range(sk_data.shape[1]):
            if np.isnan(sk_data[row_idx, col_idx]):
                self.assertTrue(np.isnan(rasl_data.iloc[row_idx, col_idx]))
            else:
                self.assertAlmostEqual(
                    sk_data[row_idx, col_idx],
                    rasl_data.iloc[row_idx, col_idx],
                    msg=(row_idx, col_idx, msg),
                )

class TestMinMaxScaler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.targets = ["pandas", "spark", "spark-with-index"]

    def test1(self):
        """
        From https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.MinMaxScaler.html
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        >>> scaler = MinMaxScaler()
        >>> print(scaler.fit(data))
        MinMaxScaler()
        >>> print(scaler.data_max_)
        [ 1. 18.]
        >>> print(scaler.transform(data))
        [[0.   0.  ]
        [0.25 0.25]
        [0.5  0.5 ]
        [1.   1.  ]]
        >>> print(scaler.transform([[2, 2]]))
        [[1.5 0. ]]
        """
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        sk_scaler = SkMinMaxScaler()
        sk_scaler.fit(data)
        sk_transformed_data = sk_scaler.transform(data)
        X = [[2, 2]]
        sk_transformed_X = sk_scaler.transform(X)

        for target in self.targets:
            data = Convert(astype=target).transform(data)
            rasl_scaler = RaslMinMaxScaler()
            rasl_scaler.fit(data)
            _check_trained_min_max_scaler(self, sk_scaler, rasl_scaler, target)
            rasl_transformed_data = rasl_scaler.transform(data)
            _check_data(self, sk_transformed_data, rasl_transformed_data, target)
            X = Convert(astype=target).transform(X)
            rasl_transformed_X = rasl_scaler.transform(X)
            _check_data(self, sk_transformed_X, rasl_transformed_X, target)


class TestOrdinalEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.targets = ["pandas", "spark", "spark-with-index"]

    def test_1(self):
        """
        From https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
        >>> from sklearn.preprocessing import OrdinalEncoder
        >>> enc = OrdinalEncoder()
        >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
        >>> enc.fit(X)
        OrdinalEncoder()
        >>> enc.categories_
        [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
        >>> enc.transform([['Female', 3], ['Male', 1]])
        array([[0., 2.],
            [1., 0.]])
        """
        sk_enc = SkOrdinalEncoder()
        X = [['Male', 1], ['Female', 3], ['Female', 2]]
        sk_enc.fit(X)
        data = [['Female', 3], ['Male', 1]]
        sk_transformed = sk_enc.transform(data)
        for target in self.targets:
            rasl_enc = RaslOrdinalEncoder()
            X = Convert(astype=target).transform(X)
            rasl_enc.fit(X)
            data = Convert(astype=target).transform(data)
            rasl_transformed = rasl_enc.transform(data)
            _check_trained_ordinal_encoder(self, sk_enc, rasl_enc, target)
            _check_data(self, sk_transformed, rasl_transformed, target)

    def test_2(self):
        """
        From https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
        >>> import numpy as np
        >>> X = [['Male', 1], ['Female', 3], ['Female', np.nan]]
        >>> enc.fit_transform(X)
        array([[ 1.,  0.],
            [ 0.,  1.],
            [ 0., nan]])
        """
        sk_enc = SkOrdinalEncoder()
        X = [['Male', 1], ['Female', 3], ['Female', np.nan]]
        sk_transformed = sk_enc.fit_transform(X)
        for target in self.targets:
            rasl_enc = RaslOrdinalEncoder()
            X = Convert(astype=target).transform(X)
            rasl_transformed = rasl_enc.fit_transform(X)
            _check_trained_ordinal_encoder(self, sk_enc, rasl_enc, target)
            if target.startswith("spark"):
                continue  # XXX issue with NaN
            _check_data(self, sk_transformed, rasl_transformed, target)

    def test_3(self):
        """
        From https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
        >>> enc.set_params(encoded_missing_value=-1).fit_transform(X)
        array([[ 1.,  0.],
            [ 0.,  1.],
            [ 0., -1.]])
        """
        pass # XXX encoded_missing_value is not implemented


class TestSelectKBest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.targets = ["pandas", "spark-with-index"] # XXX "spark" is not supported by SelectKBest

    def test1(self):
        """
        From https://scikit-learn.org/1.1/modules/generated/sklearn.feature_selection.SelectKBest.html
        >>> from sklearn.datasets import load_digits
        >>> from sklearn.feature_selection import SelectKBest, chi2
        >>> X, y = load_digits(return_X_y=True)
        >>> X.shape
        (1797, 64)
        >>> X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
        >>> X_new.shape
        (1797, 20)
        """

        X, y = load_digits(return_X_y=True, as_frame=True)
        sk_X_new = SkSelectKBest(k=20).fit_transform(X, y) # XXX chi2 is not supported in RASL
        for target in self.targets:
            X = Convert(astype=target).transform(X)
            y = Convert(astype=target).transform(y)
            rasl_X_new = RaslSelectKBest(k=20).fit_transform(X, y)
            _check_data(self, sk_X_new, rasl_X_new, target)

