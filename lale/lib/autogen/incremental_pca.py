
from sklearn.decomposition.incremental_pca import IncrementalPCA as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class IncrementalPCAImpl():

    def __init__(self, n_components=None, whiten=False, copy=True, batch_size=None):
        self._hyperparams = {
            'n_components': n_components,
            'whiten': whiten,
            'copy': copy,
            'batch_size': batch_size}
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
    'description': 'inherited docstring for IncrementalPCA    Incremental principal components analysis (IPCA).',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'whiten', 'copy', 'batch_size'],
        'relevantToOptimizer': ['n_components', 'whiten', 'copy', 'batch_size'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 256,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of components to keep'},
            'whiten': {
                'type': 'boolean',
                'default': False,
                'description': 'When True (False by default) the ``components_`` vectors are divided by ``n_samples`` times ``components_`` to ensure uncorrelated outputs with unit component-wise variances'},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'If False, X will be overwritten'},
            'batch_size': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 3,
                    'maximumForOptimizer': 128,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of samples to use for each batch'},
        }}, {
        'XXX TODO XXX': 'Parameter: batch_size > only used when calling fit'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model with X, using minibatches of size batch_size.',
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
            'description': 'Training data, where n_samples is the number of samples and n_features is the number of features.'},
        'y': {
            
        }},
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply dimensionality reduction to X.',
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
            'description': 'New data, where n_samples is the number of samples and n_features is the number of features.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply dimensionality reduction to X.',
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
        'output_transform': _output_transform_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
IncrementalPCA = lale.operators.make_operator(IncrementalPCAImpl, _combined_schemas)

