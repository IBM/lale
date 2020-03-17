
from sklearn.preprocessing._discretization import KBinsDiscretizer as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class KBinsDiscretizerImpl():

    def __init__(self, n_bins=5, encode='onehot', strategy='quantile'):
        self._hyperparams = {
            'n_bins': n_bins,
            'encode': encode,
            'strategy': strategy}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for KBinsDiscretizer    Bin continuous data into intervals.',
    'allOf': [{
        'type': 'object',
        'required': ['n_bins', 'encode', 'strategy'],
        'relevantToOptimizer': [],
        'additionalProperties': False,
        'properties': {
            'n_bins': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }],
                'default': 5,
                'description': 'The number of bins to produce'},
            'encode': {
                'enum': ['onehot', 'onehot-dense', 'ordinal'],
                'default': 'onehot',
                'description': 'Method used to encode the transformed result'},
            'strategy': {
                'enum': ['uniform', 'quantile', 'kmeans'],
                'default': 'quantile',
                'description': 'Strategy used to define the widths of the bins'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fits the estimator.',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'laleType': 'Any',
            'XXX TODO XXX': 'numeric array-like, shape (n_samples, n_features)',
            'description': 'Data to be discretized.'},
        'y': {
            'laleType': 'Any',
            'XXX TODO XXX': 'ignored'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Discretizes the data.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'laleType': 'Any',
            'XXX TODO XXX': 'numeric array-like, shape (n_samples, n_features)',
            'description': 'Data to be discretized.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Data in the binned space.',
    'laleType': 'Any',
    'XXX TODO XXX': 'numeric array-like or sparse matrix',
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
        'output_transform': _output_transform_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
KBinsDiscretizer = lale.operators.make_operator(KBinsDiscretizerImpl, _combined_schemas)

