
from sklearn.preprocessing._function_transformer import FunctionTransformer as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class FunctionTransformerImpl():

    def __init__(self, func=None, inverse_func=None, validate=None, accept_sparse=False, pass_y='deprecated', check_inverse=True, kw_args=None, inv_kw_args=None):
        self._hyperparams = {
            'func': func,
            'inverse_func': inverse_func,
            'validate': validate,
            'accept_sparse': accept_sparse,
            'pass_y': pass_y,
            'check_inverse': check_inverse,
            'kw_args': kw_args,
            'inv_kw_args': inv_kw_args}
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
    'description': 'inherited docstring for FunctionTransformer    Constructs a transformer from an arbitrary callable.',
    'allOf': [{
        'type': 'object',
        'required': ['func', 'inverse_func', 'validate', 'accept_sparse', 'pass_y', 'check_inverse', 'kw_args', 'inv_kw_args'],
        'relevantToOptimizer': ['accept_sparse', 'pass_y'],
        'additionalProperties': False,
        'properties': {
            'func': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The callable to use for the transformation'},
            'inverse_func': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The callable to use for the inverse transformation'},
            'validate': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Indicate that the input X array should be checked before calling ``func``'},
            'accept_sparse': {
                'type': 'boolean',
                'default': False,
                'description': 'Indicate that func accepts a sparse matrix as input'},
            'pass_y': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'enum': ['deprecated']}],
                'default': 'deprecated',
                'description': 'Indicate that transform should forward the y argument to the inner callable'},
            'check_inverse': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to check that or ``func`` followed by ``inverse_func`` leads to the original inputs'},
            'kw_args': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Dictionary of additional keyword arguments to pass to func.'},
            'inv_kw_args': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Dictionary of additional keyword arguments to pass to inverse_func.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit transformer by checking X.',
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
            'description': 'Input array.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform X using the forward function.',
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
            'description': 'Input array.'},
        'y': {
            'laleType': 'Any',
            'XXX TODO XXX': '(ignored)',
            'description': ''},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transformed input.',
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
FunctionTransformer = lale.operators.make_operator(FunctionTransformerImpl, _combined_schemas)

