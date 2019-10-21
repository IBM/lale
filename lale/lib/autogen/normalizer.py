
from sklearn.preprocessing.data import Normalizer as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class NormalizerImpl():

    def __init__(self, norm='l2', copy=True):
        self._hyperparams = {
            'norm': norm,
            'copy': copy}

    def fit(self, X, y=None):
        self._sklearn_model = SKLModel(**self._hyperparams)
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for Normalizer    Normalize samples individually to unit norm.',
    'allOf': [{
        'type': 'object',
        'required': ['norm', 'copy'],
        'relevantToOptimizer': ['norm', 'copy'],
        'additionalProperties': False,
        'properties': {
            'norm': {
                'XXX TODO XXX': "'l1', 'l2', or 'max', optional ('l2' by default)",
                'description': 'The norm to use to normalize each non zero sample.',
                'enum': ['l2'],
                'default': 'l2'},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'set to False to perform inplace row normalization and avoid a'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Do nothing and return the estimator unchanged',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'XXX TODO XXX': 'item type'},
            'XXX TODO XXX': 'array-like'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Scale each non zero row of X to unit norm',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data to normalize, row by row. scipy.sparse matrices should be'},
        'y': {
            'XXX TODO XXX': '(ignored)',
            'description': '.. deprecated:: 0.19'},
        'copy': {
            'anyOf': [{
                'type': 'boolean'}, {
                'enum': [None]}],
            'default': None,
            'description': 'Copy the input X or not.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Scale each non zero row of X to unit norm',
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
Normalizer = lale.operators.make_operator(NormalizerImpl, _combined_schemas)

