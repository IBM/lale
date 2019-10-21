
from sklearn.preprocessing._encoders import OrdinalEncoder as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class OrdinalEncoderImpl():

    def __init__(self, categories='auto', dtype=None):
        self._hyperparams = {
            'categories': categories,
            'dtype': dtype}

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
    'description': 'inherited docstring for OrdinalEncoder    Encode categorical features as an integer array.',
    'allOf': [{
        'type': 'object',
        'required': ['categories', 'dtype'],
        'relevantToOptimizer': [],
        'additionalProperties': False,
        'properties': {
            'categories': {
                'XXX TODO XXX': "'auto' or a list of lists/arrays of values.",
                'description': 'Categories (unique values) per feature:',
                'enum': ['auto'],
                'default': 'auto'},
            'dtype': {
                'XXX TODO XXX': 'number type, default np.float64',
                'description': 'Desired dtype of output.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the OrdinalEncoder to X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data to determine the categories of each feature.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform X to ordinal codes.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data to encode.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transformed input.',
    'XXX TODO XXX': 'sparse matrix or a 2-d array',
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
OrdinalEncoder = lale.operators.make_operator(OrdinalEncoderImpl, _combined_schemas)

