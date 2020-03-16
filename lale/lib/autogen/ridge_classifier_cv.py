
from sklearn.linear_model.ridge import RidgeClassifierCV as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class RidgeClassifierCVImpl():

    def __init__(self, alphas=[0.1, 1.0, 10.0], fit_intercept=True, normalize=False, scoring=None, cv=None, class_weight='balanced', store_cv_values=False):
        self._hyperparams = {
            'alphas': alphas,
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'scoring': scoring,
            'cv': cv,
            'class_weight': class_weight,
            'store_cv_values': store_cv_values}
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
    'description': 'inherited docstring for RidgeClassifierCV    Ridge classifier with built-in cross-validation.',
    'allOf': [{
        'type': 'object',
        'required': ['alphas', 'fit_intercept', 'normalize', 'scoring', 'cv', 'class_weight', 'store_cv_values'],
        'relevantToOptimizer': ['fit_intercept', 'normalize', 'cv'],
        'additionalProperties': False,
        'properties': {
            'alphas': {
                'type': 'array',
                'items': {
                    'type': 'number'},
                'default': [0.1, 1.0, 10.0],
                'description': 'Array of alpha values to try'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to calculate the intercept for this model'},
            'normalize': {
                'type': 'boolean',
                'default': False,
                'description': 'This parameter is ignored when ``fit_intercept`` is set to False'},
            'scoring': {
                'anyOf': [{
                    'type': 'string'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'A string (see model evaluation documentation) or a scorer callable object / function with signature ``scorer(estimator, X, y)``.'},
            'cv': {
                'XXX TODO XXX': 'int, cross-validation generator or an iterable, optional',
                'description': 'Determines the cross-validation splitting strategy',
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 3,
                    'maximumForOptimizer': 4,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None},
            'class_weight': {
                'XXX TODO XXX': "dict or 'balanced', optional",
                'description': 'Weights associated with classes in the form ``{class_label: weight}``',
                'enum': ['balanced'],
                'default': 'balanced'},
            'store_cv_values': {
                'type': 'boolean',
                'default': False,
                'description': 'Flag indicating if the cross-validation values corresponding to each alpha should be stored in the ``cv_values_`` attribute (see below)'},
        }}, {
        'XXX TODO XXX': 'Parameter: store_cv_values > only compatible with cv=none (i'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the ridge classifier.',
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
            'description': 'Training vectors, where n_samples is the number of samples and n_features is the number of features.'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values'},
        'sample_weight': {
            'anyOf': [{
                'type': 'number'}, {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }],
            'description': 'Sample weight.'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class labels for samples in X.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'laleType': 'Any',
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array_like or sparse matrix, shape (n_samples, n_features)'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'Samples.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predicted class label per sample.',
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
RidgeClassifierCV = lale.operators.make_operator(RidgeClassifierCVImpl, _combined_schemas)

