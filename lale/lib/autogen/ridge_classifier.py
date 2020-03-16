
from sklearn.linear_model.ridge import RidgeClassifier as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class RidgeClassifierImpl():

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight='balanced', solver='auto', random_state=None):
        self._hyperparams = {
            'alpha': alpha,
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'copy_X': copy_X,
            'max_iter': max_iter,
            'tol': tol,
            'class_weight': class_weight,
            'solver': solver,
            'random_state': random_state}
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
    'description': 'inherited docstring for RidgeClassifier    Classifier using Ridge regression.',
    'allOf': [{
        'type': 'object',
        'required': ['alpha', 'fit_intercept', 'normalize', 'copy_X', 'max_iter', 'tol', 'class_weight', 'solver', 'random_state'],
        'relevantToOptimizer': ['alpha', 'fit_intercept', 'normalize', 'copy_X', 'max_iter', 'tol', 'solver'],
        'additionalProperties': False,
        'properties': {
            'alpha': {
                'type': 'number',
                'minimumForOptimizer': 1e-10,
                'maximumForOptimizer': 1.0,
                'distribution': 'loguniform',
                'default': 1.0,
                'description': 'Regularization strength; must be a positive float'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to calculate the intercept for this model'},
            'normalize': {
                'type': 'boolean',
                'default': False,
                'description': 'This parameter is ignored when ``fit_intercept`` is set to False'},
            'copy_X': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, X will be copied; else, it may be overwritten.'},
            'max_iter': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 10,
                    'maximumForOptimizer': 1000,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Maximum number of iterations for conjugate gradient solver'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.001,
                'description': 'Precision of the solution.'},
            'class_weight': {
                'XXX TODO XXX': "dict or 'balanced', optional",
                'description': 'Weights associated with classes in the form ``{class_label: weight}``',
                'enum': ['balanced'],
                'default': 'balanced'},
            'solver': {
                'enum': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'default': 'auto',
                'description': "Solver to use in the computational routines:  - 'auto' chooses the solver automatically based on the type of data"},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The seed of the pseudo random number generator to use when shuffling the data'},
        }}, {
        'XXX TODO XXX': 'Parameter: solver > only guaranteed on features with approximately the same scale'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit Ridge regression model.',
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
            'description': 'Training data'},
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
            'description': 'Sample weight'},
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
RidgeClassifier = lale.operators.make_operator(RidgeClassifierImpl, _combined_schemas)

