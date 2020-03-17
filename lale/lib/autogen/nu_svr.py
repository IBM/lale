
from sklearn.svm.classes import NuSVR as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class NuSVRImpl():

    def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=(- 1)):
        self._hyperparams = {
            'nu': nu,
            'C': C,
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
            'coef0': coef0,
            'shrinking': shrinking,
            'tol': tol,
            'cache_size': cache_size,
            'verbose': verbose,
            'max_iter': max_iter}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for NuSVR    Nu Support Vector Regression.',
    'allOf': [{
        'type': 'object',
        'required': ['nu', 'C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking', 'tol', 'cache_size', 'verbose', 'max_iter'],
        'relevantToOptimizer': ['kernel', 'degree', 'gamma', 'shrinking', 'tol', 'cache_size', 'max_iter'],
        'additionalProperties': False,
        'properties': {
            'nu': {
                'type': 'number',
                'default': 0.5,
                'description': 'An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors'},
            'C': {
                'type': 'number',
                'default': 1.0,
                'description': 'Penalty parameter C of the error term.'},
            'kernel': {
                'enum': ['linear', 'poly', 'sigmoid', 'rbf'],
                'default': 'rbf',
                'description': 'Specifies the kernel type to be used in the algorithm'},
            'degree': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 3,
                'distribution': 'uniform',
                'default': 3,
                'description': "Degree of the polynomial kernel function ('poly')"},
            'gamma': {
                'anyOf': [{
                    'type': 'number',
                    'forOptimizer': False}, {
                    'enum': ['auto_deprecated']}],
                'default': 'auto_deprecated',
                'description': "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'"},
            'coef0': {
                'type': 'number',
                'default': 0.0,
                'description': 'Independent term in kernel function'},
            'shrinking': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to use the shrinking heuristic.'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.001,
                'description': 'Tolerance for stopping criterion.'},
            'cache_size': {
                'type': 'number',
                'minimumForOptimizer': 0.0,
                'maximumForOptimizer': 1.0,
                'distribution': 'uniform',
                'default': 200,
                'description': 'Specify the size of the kernel cache (in MB).'},
            'verbose': {
                'type': 'boolean',
                'default': False,
                'description': 'Enable verbose output'},
            'max_iter': {
                'XXX TODO XXX': 'int, optional (default=-1)',
                'description': 'Hard limit on iterations within solver, or -1 for no limit.',
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': (- 1)},
        }}, {
        'XXX TODO XXX': "Parameter: coef0 > only significant in 'poly' and 'sigmoid'"}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the SVM model according to the given training data.',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Training vectors, where n_samples is the number of samples and n_features is the number of features'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values (class labels in classification, real numbers in regression)'},
        'sample_weight': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Per-sample weights'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform regression on samples in X.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'For kernel="precomputed", the expected shape of X is (n_samples_test, n_samples_train).'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform regression on samples in X.',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['estimator'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
NuSVR = lale.operators.make_operator(NuSVRImpl, _combined_schemas)

