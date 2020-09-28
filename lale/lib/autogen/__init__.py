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

from .ada_boost_classifier import AdaBoostClassifier
from .ada_boost_regressor import AdaBoostRegressor
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
from .decision_tree_classifier import DecisionTreeClassifier
from .decision_tree_regressor import DecisionTreeRegressor
from .dictionary_learning import DictionaryLearning
from .elastic_net import ElasticNet
from .elastic_net_cv import ElasticNetCV
from .extra_trees_classifier import ExtraTreesClassifier
from .extra_trees_regressor import ExtraTreesRegressor
from .factor_analysis import FactorAnalysis
from .fast_ica import FastICA
from .function_transformer import FunctionTransformer
from .gaussian_nb import GaussianNB
from .gaussian_process_classifier import GaussianProcessClassifier
from .gaussian_process_regressor import GaussianProcessRegressor
from .gaussian_random_projection import GaussianRandomProjection
from .gradient_boosting_classifier import GradientBoostingClassifier
from .gradient_boosting_regressor import GradientBoostingRegressor
from .huber_regressor import HuberRegressor
from .incremental_pca import IncrementalPCA
from .isomap import Isomap
from .k_bins_discretizer import KBinsDiscretizer
from .k_means import KMeans
from .k_neighbors_classifier import KNeighborsClassifier
from .k_neighbors_regressor import KNeighborsRegressor
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
from .linear_regression import LinearRegression
from .linear_svc import LinearSVC
from .linear_svr import LinearSVR
from .locally_linear_embedding import LocallyLinearEmbedding
from .logistic_regression import LogisticRegression
from .logistic_regression_cv import LogisticRegressionCV
from .max_abs_scaler import MaxAbsScaler
from .min_max_scaler import MinMaxScaler
from .mini_batch_dictionary_learning import MiniBatchDictionaryLearning
from .mini_batch_k_means import MiniBatchKMeans
from .mini_batch_sparse_pca import MiniBatchSparsePCA
from .missing_indicator import MissingIndicator
from .mlp_classifier import MLPClassifier
from .mlp_regressor import MLPRegressor
from .multi_label_binarizer import MultiLabelBinarizer
from .multi_task_elastic_net import MultiTaskElasticNet
from .multi_task_elastic_net_cv import MultiTaskElasticNetCV
from .multi_task_lasso import MultiTaskLasso
from .multi_task_lasso_cv import MultiTaskLassoCV
from .multinomial_nb import MultinomialNB
from .nearest_centroid import NearestCentroid
from .nmf import NMF
from .normalizer import Normalizer
from .nu_svc import NuSVC
from .nu_svr import NuSVR
from .nystroem import Nystroem
from .one_hot_encoder import OneHotEncoder
from .ordinal_encoder import OrdinalEncoder
from .orthogonal_matching_pursuit import OrthogonalMatchingPursuit
from .orthogonal_matching_pursuit_cv import OrthogonalMatchingPursuitCV
from .passive_aggressive_classifier import PassiveAggressiveClassifier
from .passive_aggressive_regressor import PassiveAggressiveRegressor
from .pca import PCA
from .perceptron import Perceptron
from .pls_canonical import PLSCanonical
from .pls_regression import PLSRegression
from .plssvd import PLSSVD
from .polynomial_features import PolynomialFeatures
from .power_transformer import PowerTransformer
from .quadratic_discriminant_analysis import QuadraticDiscriminantAnalysis
from .quantile_transformer import QuantileTransformer
from .radius_neighbors_classifier import RadiusNeighborsClassifier
from .radius_neighbors_regressor import RadiusNeighborsRegressor
from .random_forest_classifier import RandomForestClassifier
from .random_forest_regressor import RandomForestRegressor
from .random_trees_embedding import RandomTreesEmbedding
from .ransac_regressor import RANSACRegressor
from .rbf_sampler import RBFSampler
from .ridge import Ridge
from .ridge_classifier import RidgeClassifier
from .ridge_classifier_cv import RidgeClassifierCV
from .ridge_cv import RidgeCV
from .robust_scaler import RobustScaler
from .sgd_classifier import SGDClassifier
from .sgd_regressor import SGDRegressor
from .simple_imputer import SimpleImputer
from .skewed_chi2_sampler import SkewedChi2Sampler
from .sparse_pca import SparsePCA
from .sparse_random_projection import SparseRandomProjection
from .standard_scaler import StandardScaler
from .svc import SVC
from .svr import SVR
from .theil_sen_regressor import TheilSenRegressor
from .transformed_target_regressor import TransformedTargetRegressor
from .truncated_svd import TruncatedSVD
