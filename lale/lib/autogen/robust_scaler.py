
from sklearn.preprocessing.data import RobustScaler as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class RobustScalerImpl():

    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True):
        self._hyperparams = {
            'with_centering': with_centering,
            'with_scaling': with_scaling,
            'quantile_range': quantile_range,
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
    'description': 'inherited docstring for RobustScaler    Scale features using statistics that are robust to outliers.',
    'allOf': [{
        'type': 'object',
        'required': ['with_centering', 'with_scaling', 'quantile_range', 'copy'],
        'relevantToOptimizer': ['with_centering', 'with_scaling', 'copy'],
        'additionalProperties': False,
        'properties': {
            'with_centering': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, center the data before scaling'},
            'with_scaling': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, scale the data to interquartile range.'},
            'quantile_range': {
                'XXX TODO XXX': 'tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0',
                'description': 'Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR Quantile range used to calculate ``scale_``',
                'type': 'array',
                'laleType': 'tuple',
                'default': (25.0, 75.0)},
            'copy': {
                'XXX TODO XXX': 'boolean, optional, default is True',
                'description': 'If False, try to avoid a copy and do inplace scaling instead',
                'type': 'boolean',
                'default': True},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute the median and quantiles to be used for scaling.',
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
            'description': 'The data used to compute the median and quantiles used for later scaling along the features axis.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Center and scale the data.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'laleType': 'Any',
                'XXX TODO XXX': 'item type'},
            'XXX TODO XXX': '{array-like, sparse matrix}',
            'description': 'The data used to scale along the specified axis.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Center and scale the data.',
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
RobustScaler = lale.operators.make_operator(RobustScalerImpl, _combined_schemas)

