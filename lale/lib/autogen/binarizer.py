
from sklearn.preprocessing.data import Binarizer as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class BinarizerImpl():

    def __init__(self, threshold=0.0, copy=True):
        self._hyperparams = {
            'threshold': threshold,
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
    'description': 'inherited docstring for Binarizer    Binarize data (set feature values to 0 or 1) according to a threshold',
    'allOf': [{
        'type': 'object',
        'required': ['threshold', 'copy'],
        'relevantToOptimizer': ['copy'],
        'additionalProperties': False,
        'properties': {
            'threshold': {
                'XXX TODO XXX': 'float, optional (0.0 by default)',
                'description': 'Feature values below or equal to this are replaced by 0, above it by 1',
                'type': 'number',
                'default': 0.0},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'set to False to perform inplace binarization and avoid a copy (if the input is already a numpy array or a scipy.sparse CSR matrix).'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Do nothing and return the estimator unchanged',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'laleType': 'Any',
                'XXX TODO XXX': 'item type'},
            'XXX TODO XXX': 'array-like'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Binarize each element of X',
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
            'description': 'The data to binarize, element by element'},
        'y': {
            'laleType': 'Any',
            'XXX TODO XXX': '(ignored)',
            'description': ''},
        'copy': {
            'type': 'boolean',
            'description': 'Copy the input X or not.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Binarize each element of X',
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
Binarizer = lale.operators.make_operator(BinarizerImpl, _combined_schemas)

