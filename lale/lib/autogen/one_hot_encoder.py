
from sklearn.preprocessing._encoders import OneHotEncoder as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class OneHotEncoderImpl():

    def __init__(self, categories=None, sparse=True, dtype=None, handle_unknown='error', n_values=None, categorical_features=None):
        self._hyperparams = {
            'categories': categories,
            'sparse': sparse,
            'dtype': dtype,
            'handle_unknown': handle_unknown,
            'n_values': n_values,
            'categorical_features': categorical_features}
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
    'description': 'inherited docstring for OneHotEncoder    Encode categorical integer features as a one-hot numeric array.',
    'allOf': [{
        'type': 'object',
        'required': ['categories', 'sparse', 'dtype', 'handle_unknown', 'n_values', 'categorical_features'],
        'relevantToOptimizer': ['sparse'],
        'additionalProperties': False,
        'properties': {
            'categories': {
                'XXX TODO XXX': "'auto' or a list of lists/arrays of values, default='auto'.",
                'description': "Categories (unique values) per feature:  - 'auto' : Determine categories automatically from the training data",
                'enum': [None],
                'default': None},
            'sparse': {
                'type': 'boolean',
                'default': True,
                'description': 'Will return sparse matrix if set True else will return an array.'},
            'dtype': {
                'laleType': 'Any',
                'XXX TODO XXX': 'number type, default=np.float',
                'description': 'Desired dtype of output.'},
            'handle_unknown': {
                'XXX TODO XXX': "'error' or 'ignore', default='error'.",
                'description': 'Whether to raise an error or ignore if an unknown categorical feature is present during transform (default is to raise)',
                'enum': ['error'],
                'default': 'error'},
            'n_values': {
                'XXX TODO XXX': "'auto', int or array of ints, default='auto'",
                'description': 'Number of values per feature',
                'enum': [None],
                'default': None},
            'categorical_features': {
                'XXX TODO XXX': "'all' or array of indices or mask, default='all'",
                'description': 'Specify what features are treated as categorical',
                'enum': [None],
                'default': None},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit OneHotEncoder to X.',
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
            'description': 'The data to determine the categories of each feature.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform X using one-hot encoding.',
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
            'description': 'The data to encode.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transformed input.',
    'laleType': 'Any',
    'XXX TODO XXX': 'sparse matrix if sparse=True else a 2-d array',
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
OneHotEncoder = lale.operators.make_operator(OneHotEncoderImpl, _combined_schemas)

