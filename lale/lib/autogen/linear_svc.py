
from sklearn.svm.classes import LinearSVC as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LinearSVCImpl():

    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight='balanced', verbose=0, random_state=None, max_iter=1000):
        self._hyperparams = {
            'penalty': penalty,
            'loss': loss,
            'dual': dual,
            'tol': tol,
            'C': C,
            'multi_class': multi_class,
            'fit_intercept': fit_intercept,
            'intercept_scaling': intercept_scaling,
            'class_weight': class_weight,
            'verbose': verbose,
            'random_state': random_state,
            'max_iter': max_iter}
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
    'description': 'inherited docstring for LinearSVC    Linear Support Vector Classification.',
    'allOf': [{
        'type': 'object',
        'required': ['penalty', 'loss', 'dual', 'tol', 'C', 'multi_class', 'fit_intercept', 'intercept_scaling', 'class_weight', 'verbose', 'random_state', 'max_iter'],
        'relevantToOptimizer': ['penalty', 'loss', 'dual', 'tol', 'multi_class', 'fit_intercept', 'intercept_scaling', 'max_iter'],
        'additionalProperties': False,
        'properties': {
            'penalty': {
                'XXX TODO XXX': "string, 'l1' or 'l2' (default='l2')",
                'description': 'Specifies the norm used in the penalization',
                'enum': ['l2', 'squared_hinge'],
                'default': 'l2'},
            'loss': {
                'XXX TODO XXX': "string, 'hinge' or 'squared_hinge' (default='squared_hinge')",
                'description': 'Specifies the loss function',
                'enum': ['epsilon_insensitive', 'hinge', 'l2', 'squared_epsilon_insensitive', 'squared_hinge'],
                'default': 'squared_hinge'},
            'dual': {
                'type': 'boolean',
                'default': True,
                'description': 'Select the algorithm to either solve the dual or primal optimization problem'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'Tolerance for stopping criteria.'},
            'C': {
                'type': 'number',
                'default': 1.0,
                'description': 'Penalty parameter C of the error term.'},
            'multi_class': {
                'XXX TODO XXX': "string, 'ovr' or 'crammer_singer' (default='ovr')",
                'description': 'Determines the multi-class strategy if `y` contains more than two classes',
                'enum': ['auto', 'crammer_singer', 'liblinear', 'ovr'],
                'default': 'ovr'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to calculate the intercept for this model'},
            'intercept_scaling': {
                'type': 'number',
                'minimumForOptimizer': 0.0,
                'maximumForOptimizer': 1.0,
                'distribution': 'uniform',
                'default': 1,
                'description': 'When self.fit_intercept is True, instance vector x becomes ``[x, self.intercept_scaling]``, i.e'},
            'class_weight': {
                'enum': ['dict', 'balanced'],
                'default': 'balanced',
                'description': 'Set the parameter C of class i to ``class_weight[i]*C`` for SVC'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'Enable verbose output'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The seed of the pseudo random number generator to use when shuffling the data for the dual coordinate descent (if ``dual=True``)'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 1000,
                'description': 'The maximum number of iterations to be run.'},
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
            'description': 'Training vector, where n_samples in the number of samples and n_features is the number of features.'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target vector relative to X'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'default': None,
            'description': 'Array of weights that are assigned to individual samples'},
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
LinearSVC = lale.operators.make_operator(LinearSVCImpl, _combined_schemas)

