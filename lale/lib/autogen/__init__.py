""" Lale autogen schemas

The JSON schemas of the operators defined in this module were automatically generated from the source code of 115 scikit-learn operators.
The resulting schemas are all valid and usable to build Lale pipelines.

The following paper describes the schema extractor::

  @InProceedings{baudart_et_al_2020,
    title = "Mining Documentation to Extract Hyperparameter Schemas",
    author = "Baudart, Guillaume and Kirchner, Peter and Hirzel, Martin and Kate, Kiran",
    booktitle = "ICML Workshop on Automated Machine Learning (AutoML@ICML)",
    year = 2020,
    url = "https://arxiv.org/abs/2006.16984" }

"""

from lale.lib.sklearn.ada_boost_classifier import AdaBoostClassifier
from lale.lib.sklearn.ada_boost_regressor import AdaBoostRegressor
from lale.lib.sklearn.decision_tree_classifier import DecisionTreeClassifier
from lale.lib.sklearn.decision_tree_regressor import DecisionTreeRegressor
from lale.lib.sklearn.extra_trees_classifier import ExtraTreesClassifier
from lale.lib.sklearn.extra_trees_regressor import ExtraTreesRegressor
from lale.lib.sklearn.function_transformer import FunctionTransformer
from lale.lib.sklearn.gaussian_nb import GaussianNB
from lale.lib.sklearn.gradient_boosting_classifier import GradientBoostingClassifier
from lale.lib.sklearn.gradient_boosting_regressor import GradientBoostingRegressor
from lale.lib.sklearn.isomap import Isomap
from lale.lib.sklearn.k_means import KMeans
from lale.lib.sklearn.k_neighbors_classifier import KNeighborsClassifier
from lale.lib.sklearn.k_neighbors_regressor import KNeighborsRegressor
from lale.lib.sklearn.linear_regression import LinearRegression
from lale.lib.sklearn.linear_svc import LinearSVC
from lale.lib.sklearn.linear_svr import LinearSVR
from lale.lib.sklearn.logistic_regression import LogisticRegression
from lale.lib.sklearn.min_max_scaler import MinMaxScaler
from lale.lib.sklearn.missing_indicator import MissingIndicator
from lale.lib.sklearn.mlp_classifier import MLPClassifier
from lale.lib.sklearn.multinomial_nb import MultinomialNB
from lale.lib.sklearn.nmf import NMF
from lale.lib.sklearn.normalizer import Normalizer
from lale.lib.sklearn.nystroem import Nystroem
from lale.lib.sklearn.one_hot_encoder import OneHotEncoder
from lale.lib.sklearn.ordinal_encoder import OrdinalEncoder
from lale.lib.sklearn.passive_aggressive_classifier import PassiveAggressiveClassifier
from lale.lib.sklearn.pca import PCA
from lale.lib.sklearn.polynomial_features import PolynomialFeatures
from lale.lib.sklearn.quadratic_discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
)
from lale.lib.sklearn.quantile_transformer import QuantileTransformer
from lale.lib.sklearn.random_forest_classifier import RandomForestClassifier
from lale.lib.sklearn.random_forest_regressor import RandomForestRegressor
from lale.lib.sklearn.ridge import Ridge
from lale.lib.sklearn.ridge_classifier import RidgeClassifier
from lale.lib.sklearn.robust_scaler import RobustScaler
from lale.lib.sklearn.sgd_classifier import SGDClassifier
from lale.lib.sklearn.sgd_regressor import SGDRegressor
from lale.lib.sklearn.simple_imputer import SimpleImputer
from lale.lib.sklearn.standard_scaler import StandardScaler
from lale.lib.sklearn.svc import SVC
from lale.lib.sklearn.svr import SVR

from .additive_chi2_sampler import AdditiveChi2Sampler
from .ard_regression import ARDRegression
from .bayesian_ridge import BayesianRidge
from .bernoulli_nb import BernoulliNB
from .bernoulli_rbm import BernoulliRBM
from .binarizer import Binarizer
from .birch import Birch
from .calibrated_classifier_cv import CalibratedClassifierCV
from .cca import CCA
from .complement_nb import ComplementNB
from .dictionary_learning import DictionaryLearning
from .elastic_net import ElasticNet
from .elastic_net_cv import ElasticNetCV
from .factor_analysis import FactorAnalysis
from .fast_ica import FastICA
from .gaussian_process_classifier import GaussianProcessClassifier
from .gaussian_process_regressor import GaussianProcessRegressor
from .gaussian_random_projection import GaussianRandomProjection
from .huber_regressor import HuberRegressor
from .incremental_pca import IncrementalPCA
from .k_bins_discretizer import KBinsDiscretizer
from .kernel_pca import KernelPCA
from .kernel_ridge import KernelRidge
from .label_binarizer import LabelBinarizer
from .label_encoder import LabelEncoder
from .label_propagation import LabelPropagation
from .label_spreading import LabelSpreading
from .lars import Lars
from .lars_cv import LarsCV
from .lasso import Lasso
from .lasso_cv import LassoCV
from .lasso_lars import LassoLars
from .lasso_lars_cv import LassoLarsCV
from .lasso_lars_ic import LassoLarsIC
from .latent_dirichlet_allocation import LatentDirichletAllocation
from .linear_discriminant_analysis import LinearDiscriminantAnalysis
from .locally_linear_embedding import LocallyLinearEmbedding
from .logistic_regression_cv import LogisticRegressionCV
from .max_abs_scaler import MaxAbsScaler
from .mini_batch_dictionary_learning import MiniBatchDictionaryLearning
from .mini_batch_k_means import MiniBatchKMeans
from .mini_batch_sparse_pca import MiniBatchSparsePCA
from .mlp_regressor import MLPRegressor
from .multi_label_binarizer import MultiLabelBinarizer
from .multi_task_elastic_net import MultiTaskElasticNet
from .multi_task_elastic_net_cv import MultiTaskElasticNetCV
from .multi_task_lasso import MultiTaskLasso
from .multi_task_lasso_cv import MultiTaskLassoCV
from .nearest_centroid import NearestCentroid
from .nu_svc import NuSVC
from .nu_svr import NuSVR
from .orthogonal_matching_pursuit import OrthogonalMatchingPursuit
from .orthogonal_matching_pursuit_cv import OrthogonalMatchingPursuitCV
from .passive_aggressive_regressor import PassiveAggressiveRegressor
from .perceptron import Perceptron
from .pls_canonical import PLSCanonical
from .pls_regression import PLSRegression
from .plssvd import PLSSVD
from .power_transformer import PowerTransformer
from .radius_neighbors_classifier import RadiusNeighborsClassifier
from .radius_neighbors_regressor import RadiusNeighborsRegressor
from .random_trees_embedding import RandomTreesEmbedding
from .ransac_regressor import RANSACRegressor
from .rbf_sampler import RBFSampler
from .ridge_classifier_cv import RidgeClassifierCV
from .ridge_cv import RidgeCV
from .skewed_chi2_sampler import SkewedChi2Sampler
from .sparse_pca import SparsePCA
from .sparse_random_projection import SparseRandomProjection
from .theil_sen_regressor import TheilSenRegressor
from .transformed_target_regressor import TransformedTargetRegressor
from .truncated_svd import TruncatedSVD
