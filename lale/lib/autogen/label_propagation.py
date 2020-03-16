
from sklearn.semi_supervised.label_propagation import LabelPropagation as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LabelPropagationImpl():

    def __init__(self, kernel='rbf', gamma=20, n_neighbors=7, alpha=None, max_iter=1000, tol=0.001, n_jobs=None):
        self._hyperparams = {
            'kernel': kernel,
            'gamma': gamma,
            'n_neighbors': n_neighbors,
            'alpha': alpha,
            'max_iter': max_iter,
            'tol': tol,
            'n_jobs': n_jobs}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)

    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for LabelPropagation    Label Propagation classifier',
    'allOf': [{
        'type': 'object',
        'required': ['kernel', 'gamma', 'n_neighbors', 'alpha', 'max_iter', 'tol', 'n_jobs'],
        'relevantToOptimizer': ['kernel', 'gamma', 'n_neighbors', 'max_iter', 'tol'],
        'additionalProperties': False,
        'properties': {
            'kernel': {
                'enum': ['knn', 'rbf', 'callable'],
                'default': 'rbf',
                'description': 'String identifier for kernel function to use or the kernel function itself'},
            'gamma': {
                'type': 'number',
                'minimumForOptimizer': 0.0,
                'maximumForOptimizer': 20.0,
                'distribution': 'uniform',
                'default': 20,
                'description': 'Parameter for rbf kernel'},
            'n_neighbors': {
                'XXX TODO XXX': 'integer > 0',
                'description': 'Parameter for knn kernel',
                'type': 'integer',
                'minimumForOptimizer': 5,
                'maximumForOptimizer': 20,
                'distribution': 'uniform',
                'default': 7},
            'alpha': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Clamping factor'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 1000,
                'description': 'Change maximum number of iterations allowed'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.001,
                'description': 'Convergence tolerance: threshold to consider the system at steady state'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of parallel jobs to run'},
        }}, {
        'XXX TODO XXX': "Parameter: kernel > only 'rbf' and 'knn' strings are valid inputs"}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit a semi-supervised label propagation model based',
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
            'description': 'A {n_samples by n_samples} size matrix will be created from this'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'n_labeled_samples (unlabeled points are marked as -1) All unlabeled samples will be transductively assigned labels'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Performs inductive inference across the model.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predictions for input data',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict probability for each possible outcome.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }},
    },
}
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Normalized probability distributions across class labels',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'},
    },
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
        'output_predict': _output_predict_schema,
        'input_predict_proba': _input_predict_proba_schema,
        'output_predict_proba': _output_predict_proba_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
LabelPropagation = lale.operators.make_operator(LabelPropagationImpl, _combined_schemas)

