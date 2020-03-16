
from sklearn.preprocessing.data import MaxAbsScaler as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class MaxAbsScalerImpl():

    def __init__(self, copy=True):
        self._hyperparams = {
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
    'description': 'inherited docstring for MaxAbsScaler    Scale each feature by its maximum absolute value.',
    'allOf': [{
        'type': 'object',
        'required': ['copy'],
        'relevantToOptimizer': ['copy'],
        'additionalProperties': False,
        'properties': {
            'copy': {
                'XXX TODO XXX': 'boolean, optional, default is True',
                'description': 'Set to False to perform inplace scaling and avoid a copy (if the input is already a numpy array).',
                'type': 'boolean',
                'default': True},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute the maximum absolute value to be used for later scaling.',
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
            'description': 'The data used to compute the per-feature minimum and maximum used for later scaling along the features axis.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Scale the data',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'laleType': 'Any',
                'XXX TODO XXX': 'item type'},
            'XXX TODO XXX': '{array-like, sparse matrix}',
            'description': 'The data that should be scaled.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Scale the data',
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
MaxAbsScaler = lale.operators.make_operator(MaxAbsScalerImpl, _combined_schemas)

