
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LinearDiscriminantAnalysisImpl():

    def __init__(self, solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001):
        self._hyperparams = {
            'solver': solver,
            'shrinkage': shrinkage,
            'priors': priors,
            'n_components': n_components,
            'store_covariance': store_covariance,
            'tol': tol}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)

    def predict(self, X):
        return self._sklearn_model.predict(X)

    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for LinearDiscriminantAnalysis    Linear Discriminant Analysis',
    'allOf': [{
        'type': 'object',
        'required': ['solver', 'shrinkage', 'priors', 'n_components', 'store_covariance', 'tol'],
        'relevantToOptimizer': ['solver', 'n_components', 'tol'],
        'additionalProperties': False,
        'properties': {
            'solver': {
                'enum': ['lsqr', 'svd'],
                'default': 'svd',
                'description': "Solver to use, possible values:   - 'svd': Singular value decomposition (default)"},
            'shrinkage': {
                'anyOf': [{
                    'type': 'string'}, {
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Shrinkage parameter, possible values:   - None: no shrinkage (default)'},
            'priors': {
                'XXX TODO XXX': 'array, optional, shape (n_classes,)',
                'description': 'Class priors.',
                'enum': [None],
                'default': None},
            'n_components': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 256,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of components (< n_classes - 1) for dimensionality reduction.'},
            'store_covariance': {
                'type': 'boolean',
                'default': False,
                'description': "Additionally compute class covariance matrix (default False), used only in 'svd' solver"},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'Threshold used for rank estimation in SVD solver'},
        }}, {
        'description': "shrinkage, only with 'lsqr' and 'eigen' solvers",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'shrinkage': {
                    'enum': [None]},
            }}, {
            'type': 'object',
            'properties': {
                'solvers': {
                    'enum': ['lsqr', 'eigen']},
            }}]}, {
        'description': "store_covariance, only in 'svd' solver",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'store_covariance': {
                    'enum': [False]},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['svd']},
            }}]}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit LinearDiscriminantAnalysis model according to the given',
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
            'description': 'Training data.'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Project data to maximize class separation.',
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
            'description': 'Input data.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transformed data.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class labels for samples in X.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'laleType': 'Any',
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array_like or sparse matrix, shape (n_samples, n_features)'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'Samples.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predicted class label per sample.',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Estimate probability.',
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
            'description': 'Input data.'},
    },
}
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Estimated probabilities.',
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
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema,
        'input_predict_proba': _input_predict_proba_schema,
        'output_predict_proba': _output_predict_proba_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
LinearDiscriminantAnalysis = lale.operators.make_operator(LinearDiscriminantAnalysisImpl, _combined_schemas)

