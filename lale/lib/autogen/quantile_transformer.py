
from sklearn.preprocessing.data import QuantileTransformer as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class QuantileTransformerImpl():

    def __init__(self, n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False, subsample=100000, random_state=None, copy=True):
        self._hyperparams = {
            'n_quantiles': n_quantiles,
            'output_distribution': output_distribution,
            'ignore_implicit_zeros': ignore_implicit_zeros,
            'subsample': subsample,
            'random_state': random_state,
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
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for QuantileTransformer    Transform features using quantiles information.',
    'allOf': [{
        'type': 'object',
        'required': ['n_quantiles', 'output_distribution', 'ignore_implicit_zeros', 'subsample', 'random_state', 'copy'],
        'relevantToOptimizer': ['n_quantiles', 'output_distribution', 'subsample', 'copy'],
        'additionalProperties': False,
        'properties': {
            'n_quantiles': {
                'type': 'integer',
                'minimumForOptimizer': 1000,
                'maximumForOptimizer': 1001,
                'distribution': 'uniform',
                'default': 1000,
                'description': 'Number of quantiles to be computed'},
            'output_distribution': {
                'enum': ['normal', 'uniform'],
                'default': 'uniform',
                'description': 'Marginal distribution for the transformed data'},
            'ignore_implicit_zeros': {
                'type': 'boolean',
                'default': False,
                'description': 'Only applies to sparse matrices'},
            'subsample': {
                'type': 'integer',
                'minimumForOptimizer': 1,
                'maximumForOptimizer': 100000,
                'distribution': 'uniform',
                'default': 100000,
                'description': 'Maximum number of samples used to estimate the quantiles for computational efficiency'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random'},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'Set to False to perform inplace transformation and avoid a copy (if the input is already a numpy array).'},
        }}, {
        'XXX TODO XXX': 'Parameter: ignore_implicit_zeros > only applies to sparse matrices'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute the quantiles used for transforming.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'laleType': 'Any',
            'XXX TODO XXX': 'ndarray or sparse matrix, shape (n_samples, n_features)',
            'description': 'The data used to scale along the features axis'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Feature-wise transformation of the data.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'laleType': 'Any',
            'XXX TODO XXX': 'ndarray or sparse matrix, shape (n_samples, n_features)',
            'description': 'The data used to scale along the features axis'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The projected data.',
    'laleType': 'Any',
    'XXX TODO XXX': 'ndarray or sparse matrix, shape (n_samples, n_features)',
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
QuantileTransformer = lale.operators.make_operator(QuantileTransformerImpl, _combined_schemas)

