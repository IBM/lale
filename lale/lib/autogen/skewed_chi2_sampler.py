
from sklearn.kernel_approximation import SkewedChi2Sampler as Op
import lale.helpers
import lale.operators
import lale.docstrings
from numpy import nan, inf

class SkewedChi2SamplerImpl():

    def __init__(self, skewedness=1.0, n_components=100, random_state=None):
        self._hyperparams = {
            'skewedness': skewedness,
            'n_components': n_components,
            'random_state': random_state}
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for SkewedChi2Sampler    Approximates feature map of the "skewed chi-squared" kernel by Monte',
    'allOf': [{
        'type': 'object',
        'required': ['skewedness', 'n_components', 'random_state'],
        'relevantToOptimizer': ['n_components'],
        'additionalProperties': False,
        'properties': {
            'skewedness': {
                'type': 'number',
                'default': 1.0,
                'description': '"skewedness" parameter of the kernel'},
            'n_components': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 256,
                'distribution': 'uniform',
                'default': 100,
                'description': 'number of Monte Carlo samples per original feature'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model with X.',
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
            'description': 'Training data, where n_samples in the number of samples and n_features is the number of features.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply the approximate feature map to X.',
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
            'description': 'New data, where n_samples in the number of samples and n_features is the number of features'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply the approximate feature map to X.',
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
    'documentation_url': 'https://scikit-learn.org/0.20/modules/generated/sklearn.kernel_approximation.SkewedChi2Sampler#sklearn-kernel_approximation-skewedchi2sampler',
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
lale.docstrings.set_docstrings(SkewedChi2SamplerImpl, _combined_schemas)
SkewedChi2Sampler = lale.operators.make_operator(SkewedChi2SamplerImpl, _combined_schemas)

