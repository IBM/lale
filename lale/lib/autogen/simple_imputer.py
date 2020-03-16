
from sklearn.impute import SimpleImputer as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class SimpleImputerImpl():

    def __init__(self, missing_values='nan', strategy='mean', fill_value=None, verbose=0, copy=True):
        self._hyperparams = {
            'missing_values': missing_values,
            'strategy': strategy,
            'fill_value': fill_value,
            'verbose': verbose,
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
    'description': 'inherited docstring for SimpleImputer    Imputation transformer for completing missing values.',
    'allOf': [{
        'type': 'object',
        'required': ['missing_values', 'strategy', 'fill_value', 'verbose', 'copy'],
        'relevantToOptimizer': [],
        'additionalProperties': False,
        'properties': {
            'missing_values': {
                'XXX TODO XXX': 'number, string, np.nan (default) or None',
                'description': 'The placeholder for the missing values',
                'type': 'number',
                'default': nan},
            'strategy': {
                'type': 'string',
                'default': 'mean',
                'description': 'The imputation strategy'},
            'fill_value': {
                'XXX TODO XXX': 'string or numerical value, optional (default=None)',
                'description': 'When strategy == "constant", fill_value is used to replace all occurrences of missing_values',
                'enum': [None],
                'default': None},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'Controls the verbosity of the imputer.'},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, a copy of X will be created'},
        }}, {
        'XXX TODO XXX': 'Parameter: strategy > only be used with numeric data'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the imputer on X.',
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
            'description': 'Input data, where ``n_samples`` is the number of samples and ``n_features`` is the number of features.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Impute all missing values in X.',
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
            'description': 'The input data to complete.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Impute all missing values in X.',
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
SimpleImputer = lale.operators.make_operator(SimpleImputerImpl, _combined_schemas)

