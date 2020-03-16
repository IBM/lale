
from sklearn.compose._target import TransformedTargetRegressor as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class TransformedTargetRegressorImpl():

    def __init__(self, regressor=None, transformer=None, func=None, inverse_func=None, check_inverse=True):
        self._hyperparams = {
            'regressor': regressor,
            'transformer': transformer,
            'func': func,
            'inverse_func': inverse_func,
            'check_inverse': check_inverse}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for TransformedTargetRegressor    Meta-estimator to regress on a transformed target.',
    'allOf': [{
        'type': 'object',
        'required': ['regressor', 'transformer', 'func', 'inverse_func', 'check_inverse'],
        'relevantToOptimizer': [],
        'additionalProperties': False,
        'properties': {
            'regressor': {
                'XXX TODO XXX': 'object, default=LinearRegression()',
                'description': 'Regressor object such as derived from ``RegressorMixin``',
                'enum': [None],
                'default': None},
            'transformer': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Estimator object such as derived from ``TransformerMixin``'},
            'func': {
                'XXX TODO XXX': 'function, optional',
                'description': 'Function to apply to ``y`` before passing to ``fit``',
                'enum': [None],
                'default': None},
            'inverse_func': {
                'XXX TODO XXX': 'function, optional',
                'description': 'Function to apply to the prediction of the regressor',
                'enum': [None],
                'default': None},
            'check_inverse': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to check that ``transform`` followed by ``inverse_transform`` or ``func`` followed by ``inverse_func`` leads to the original targets.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model according to the given training data.',
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
            'description': 'Training vector, where n_samples is the number of samples and n_features is the number of features.'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values.'},
        'sample_weight': {
            'laleType': 'Any',
            'XXX TODO XXX': 'array-like, shape (n_samples,) optional',
            'description': 'Array of weights that are assigned to individual samples'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the base regressor, applying inverse.',
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
            'description': 'Samples.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predicted values.',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['estimator'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
TransformedTargetRegressor = lale.operators.make_operator(TransformedTargetRegressorImpl, _combined_schemas)

