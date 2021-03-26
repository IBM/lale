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

"""
Schema-enhanced versions of some of the operators from `scikit-learn`_ to enable hyperparameter tuning.

.. _`scikit-learn`: https://scikit-learn.org

Operators
=========

Classifiers:

* lale.lib.sklearn. `AdaBoostClassifier`_
* lale.lib.sklearn. `BaggingClassifier`_
* lale.lib.sklearn. `DecisionTreeClassifier`_
* lale.lib.sklearn. `DummyClassifier`_
* lale.lib.sklearn. `ExtraTreesClassifier`_
* lale.lib.sklearn. `GaussianNB`_
* lale.lib.sklearn. `GradientBoostingClassifier`_
* lale.lib.sklearn. `KNeighborsClassifier`_
* lale.lib.sklearn. `LinearSVC`_
* lale.lib.sklearn. `LogisticRegression`_
* lale.lib.sklearn. `MLPClassifier`_
* lale.lib.sklearn. `MultinomialNB`_
* lale.lib.sklearn. `PassiveAggressiveClassifier`_
* lale.lib.sklearn. `RandomForestClassifier`_
* lale.lib.sklearn. `RidgeClassifier`_
* lale.lib.sklearn. `SGDClassifier`_
* lale.lib.sklearn. `SVC`_
* lale.lib.sklearn. `VotingClassifier`_

Regressors:

* lale.lib.sklearn. `AdaBoostRegressor`_
* lale.lib.sklearn. `DecisionTreeRegressor`_
* lale.lib.sklearn. `DummyRegressor`_
* lale.lib.sklearn. `ExtraTreesRegressor`_
* lale.lib.sklearn. `GradientBoostingRegressor`_
* lale.lib.sklearn. `KNeighborsRegressor`_
* lale.lib.sklearn. `LinearRegression`_
* lale.lib.sklearn. `RandomForestRegressor`_
* lale.lib.sklearn. `Ridge`_
* lale.lib.sklearn. `SGDRegressor`_
* lale.lib.sklearn. `SVR`_
* lale.lib.sklearn. `LinearSVR`_

Transformers:

* lale.lib.sklearn. `ColumnTransformer`_
* lale.lib.sklearn. `FeatureAgglomeration`_
* lale.lib.sklearn. `FunctionTransformer`_
* lale.lib.sklearn. `MinMaxScaler`_
* lale.lib.sklearn. `MissingIndicator`_
* lale.lib.sklearn. `NMF`_
* lale.lib.sklearn. `Normalizer`_
* lale.lib.sklearn. `Nystroem`_
* lale.lib.sklearn. `OneHotEncoder`_
* lale.lib.sklearn. `OrdinalEncoder`_
* lale.lib.sklearn. `PCA`_
* lale.lib.sklearn. `Pipeline`_
* lale.lib.sklearn. `PolynomialFeatures`_
* lale.lib.sklearn. `QuadraticDiscriminantAnalysis`_
* lale.lib.sklearn. `QuantileTransformer`_
* lale.lib.sklearn. `RFE`_
* lale.lib.sklearn. `RobustScaler`_
* lale.lib.sklearn. `SelectKBest`_
* lale.lib.sklearn. `SimpleImputer`_
* lale.lib.sklearn. `StandardScaler`_
* lale.lib.sklearn. `TfidfVectorizer`_
* lale.lib.sklearn. `VarianceThreshold`_

.. _`AdaBoostClassifier`: lale.lib.sklearn.ada_boost_classifier.html
.. _`AdaBoostRegressor`: lale.lib.sklearn.ada_boost_regressor.html
.. _`BaggingClassifier`: lale.lib.sklearn.bagging_classifier.html
.. _`ColumnTransformer`: lale.lib.sklearn.column_transformer.html
.. _`DecisionTreeClassifier`: lale.lib.sklearn.decision_tree_classifier.html
.. _`DecisionTreeRegressor`: lale.lib.sklearn.decision_tree_regressor.html
.. _`DummyClassifier`: lale.lib.sklearn.dummy_classifier.html
.. _`DummyRegressor`: lale.lib.sklearn.dummy_regressor.html
.. _`ExtraTreesClassifier`: lale.lib.sklearn.extra_trees_classifier.html
.. _`ExtraTreesRegressor`: lale.lib.sklearn.extra_trees_regressor.html
.. _`FeatureAgglomeration`: lale.lib.sklearn.feature_agglomeration.html
.. _`FunctionTransformer`: lale.lib.sklearn.function_transformer.html
.. _`GaussianNB`: lale.lib.sklearn.gaussian_nb.html
.. _`GradientBoostingClassifier`: lale.lib.sklearn.gradient_boosting_classifier.html
.. _`GradientBoostingRegressor`: lale.lib.sklearn.gradient_boosting_regressor.html
.. _`KNeighborsClassifier`: lale.lib.sklearn.k_neighbors_classifier.html
.. _`KNeighborsRegressor`: lale.lib.sklearn.k_neighbors_regressor.html
.. _`LinearRegression`: lale.lib.sklearn.linear_regression.html
.. _`LinearSVC`: lale.lib.sklearn.linear_svc.html
.. _`LogisticRegression`: lale.lib.sklearn.logistic_regression.html
.. _`MinMaxScaler`: lale.lib.sklearn.min_max_scaler.html
.. _`MissingIndicator`: lale.lib.sklearn.missing_indicator.html
.. _`MLPClassifier`: lale.lib.sklearn.mlp_classifier.html
.. _`MultinomialNB`: lale.lib.sklearn.multinomial_nb.html
.. _`NMF`: lale.lib.sklearn.nmf.html
.. _`Normalizer`: lale.lib.sklearn.normalizer.html
.. _`Nystroem`: lale.lib.sklearn.nystroem.html
.. _`OneHotEncoder`: lale.lib.sklearn.one_hot_encoder.html
.. _`OrdinalEncoder`: lale.lib.sklearn.ordinal_encoder.html
.. _`PassiveAggressiveClassifier`: lale.lib.sklearn.passive_aggressive_classifier.html
.. _`PCA`: lale.lib.sklearn.pca.html
.. _`Pipeline`: lale.lib.sklearn.pipeline.html
.. _`PolynomialFeatures`: lale.lib.sklearn.polynomial_features.html
.. _`QuadraticDiscriminantAnalysis`: lale.lib.sklearn.quadratic_discriminant_analysis.html
.. _`QuantileTransformer`: lale.lib.sklearn.quantile_transformer.html
.. _`RandomForestClassifier`: lale.lib.sklearn.random_forest_classifier.html
.. _`RandomForestRegressor`: lale.lib.sklearn.random_forest_regressor.html
.. _`RFE`: lale.lib.sklearn.rfe.html
.. _`Ridge`: lale.lib.sklearn.ridge.html
.. _`RidgeClassifier`: lale.lib.sklearn.ridge_classifier.html
.. _`RobustScaler`: lale.lib.sklearn.robust_scaler.html
.. _`SelectKBest`: lale.lib.sklearn.select_k_best.html
.. _`SGDClassifier`: lale.lib.sklearn.sgd_classifier.html
.. _`SGDRegressor`: lale.lib.sklearn.sgd_regressor.html
.. _`SimpleImputer`: lale.lib.sklearn.simple_imputer.html
.. _`StandardScaler`: lale.lib.sklearn.standard_scaler.html
.. _`SVC`: lale.lib.sklearn.svc.html
.. _`SVR`: lale.lib.sklearn.svr.html
.. _`LinearSVR`: lale.lib.sklearn.linear_svr.html
.. _`TfidfVectorizer`: lale.lib.sklearn.tfidf_vectorizer.html
.. _`VarianceThreshold`: lale.lib.sklearn.variance_threshold.html
.. _`VotingClassifier`: lale.lib.sklearn.voting_classifier.html
"""

from .ada_boost_classifier import AdaBoostClassifier
from .ada_boost_regressor import AdaBoostRegressor
from .bagging_classifier import BaggingClassifier
from .column_transformer import ColumnTransformer
from .decision_tree_classifier import DecisionTreeClassifier
from .decision_tree_regressor import DecisionTreeRegressor
from .dummy_classifier import DummyClassifier
from .dummy_regressor import DummyRegressor
from .extra_trees_classifier import ExtraTreesClassifier
from .extra_trees_regressor import ExtraTreesRegressor
from .feature_agglomeration import FeatureAgglomeration
from .function_transformer import FunctionTransformer
from .gaussian_nb import GaussianNB
from .gradient_boosting_classifier import GradientBoostingClassifier
from .gradient_boosting_regressor import GradientBoostingRegressor
from .k_neighbors_classifier import KNeighborsClassifier
from .k_neighbors_regressor import KNeighborsRegressor
from .linear_regression import LinearRegression
from .linear_svc import LinearSVC
from .linear_svr import LinearSVR
from .logistic_regression import LogisticRegression
from .min_max_scaler import MinMaxScaler
from .missing_indicator import MissingIndicator
from .mlp_classifier import MLPClassifier
from .multinomial_nb import MultinomialNB
from .nmf import NMF
from .normalizer import Normalizer
from .nystroem import Nystroem
from .one_hot_encoder import OneHotEncoder
from .ordinal_encoder import OrdinalEncoder
from .passive_aggressive_classifier import PassiveAggressiveClassifier
from .pca import PCA
from .pipeline import Pipeline
from .polynomial_features import PolynomialFeatures
from .quadratic_discriminant_analysis import QuadraticDiscriminantAnalysis
from .quantile_transformer import QuantileTransformer
from .random_forest_classifier import RandomForestClassifier
from .random_forest_regressor import RandomForestRegressor
from .rfe import RFE
from .ridge import Ridge
from .ridge_classifier import RidgeClassifier
from .robust_scaler import RobustScaler
from .select_k_best import SelectKBest
from .sgd_classifier import SGDClassifier
from .sgd_regressor import SGDRegressor
from .simple_imputer import SimpleImputer
from .standard_scaler import StandardScaler
from .svc import SVC
from .svr import SVR
from .tfidf_vectorizer import TfidfVectorizer
from .variance_threshold import VarianceThreshold
from .voting_classifier import VotingClassifier
