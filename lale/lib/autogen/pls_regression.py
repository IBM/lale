
from sklearn.cross_decomposition.pls_ import PLSRegression as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class PLSRegressionImpl():

    def __init__(self, n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True):
        self._hyperparams = {
            'n_components': n_components,
            'scale': scale,
            'max_iter': max_iter,
            'tol': tol,
            'copy': copy}
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
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for PLSRegression    PLS regression',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'scale', 'max_iter', 'tol', 'copy'],
        'relevantToOptimizer': ['n_components', 'scale', 'max_iter', 'tol', 'copy'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 256,
                'distribution': 'uniform',
                'default': 2,
                'description': 'Number of components to keep.'},
            'scale': {
                'type': 'boolean',
                'default': True,
                'description': 'whether to scale the data'},
            'max_iter': {
                'XXX TODO XXX': 'an integer, (default 500)',
                'description': 'the maximum number of iterations of the NIPALS inner loop (used only if algorithm="nipals")',
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 500},
            'tol': {
                'XXX TODO XXX': 'non-negative real',
                'description': 'Tolerance used in the iterative algorithm default 1e-06.',
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 1e-06},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether the deflation should be done on a copy'},
        }}, {
        'XXX TODO XXX': 'Parameter: max_iter > only if algorithm="nipals")'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit model to data.',
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
            'description': 'Training vectors, where n_samples is the number of samples and n_features is the number of predictors.'},
        'Y': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply the dimension reduction learned on the train data.',
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
            'description': 'Training vectors, where n_samples is the number of samples and n_features is the number of predictors.'},
        'Y': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.'},
        'copy': {
            'type': 'boolean',
            'default': True,
            'description': 'Whether to copy X and Y, or perform in-place normalization.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply the dimension reduction learned on the train data.',
    'laleType': 'Any',
    'XXX TODO XXX': '',
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply the dimension reduction learned on the train data.',
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
            'description': 'Training vectors, where n_samples is the number of samples and n_features is the number of predictors.'},
        'copy': {
            'type': 'boolean',
            'default': True,
            'description': 'Whether to copy X and Y, or perform in-place normalization.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply the dimension reduction learned on the train data.',
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
        'output_predict': _output_predict_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
PLSRegression = lale.operators.make_operator(PLSRegressionImpl, _combined_schemas)

