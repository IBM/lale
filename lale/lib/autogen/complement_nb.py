
from sklearn.naive_bayes import ComplementNB as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class ComplementNBImpl():

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, norm=False):
        self._hyperparams = {
            'alpha': alpha,
            'fit_prior': fit_prior,
            'class_prior': class_prior,
            'norm': norm}

    def fit(self, X, y=None):
        self._sklearn_model = SKLModel(**self._hyperparams)
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)

    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for ComplementNB    The Complement Naive Bayes classifier described in Rennie et al. (2003).',
    'allOf': [{
        'type': 'object',
        'required': ['alpha', 'fit_prior', 'class_prior', 'norm'],
        'relevantToOptimizer': [],
        'additionalProperties': False,
        'properties': {
            'alpha': {
                'type': 'number',
                'default': 1.0,
                'description': 'Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).'},
            'fit_prior': {
                'type': 'boolean',
                'default': True,
                'description': 'Only used in edge case with a single class in the training set.'},
            'class_prior': {
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }, {
                    'enum': [None]}],
                'default': None,
                'description': 'Prior probabilities of the classes. Not used.'},
            'norm': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether or not a second normalization of the weights is performed. The'},
        }}, {
        'XXX TODO XXX': 'Parameter: fit_prior > only used in edge case with a single class in the training set'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit Naive Bayes classifier according to X, y',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Training vectors, where n_samples is the number of samples and'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values.'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'default': None,
            'description': 'Weights applied to individual samples (1. for unweighted).'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform classification on an array of test vectors X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predicted target values for X',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Return probability estimates for the test vector X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }},
    },
}
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Returns the probability of the samples for each class in',
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
        'op': ['estimator'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema,
        'input_predict_proba': _input_predict_proba_schema,
        'output_predict_proba': _output_predict_proba_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
ComplementNB = lale.operators.make_operator(ComplementNBImpl, _combined_schemas)

