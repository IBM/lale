
from sklearn.linear_model.logistic import LogisticRegression as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LogisticRegressionImpl():

    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight='balanced', random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=None):
        self._hyperparams = {
            'penalty': penalty,
            'dual': dual,
            'tol': tol,
            'C': C,
            'fit_intercept': fit_intercept,
            'intercept_scaling': intercept_scaling,
            'class_weight': class_weight,
            'random_state': random_state,
            'solver': solver,
            'max_iter': max_iter,
            'multi_class': multi_class,
            'verbose': verbose,
            'warm_start': warm_start,
            'n_jobs': n_jobs}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
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
    'description': 'inherited docstring for LogisticRegression    Logistic Regression (aka logit, MaxEnt) classifier.',
    'allOf': [{
        'type': 'object',
        'required': ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling', 'class_weight', 'random_state', 'solver', 'max_iter', 'multi_class', 'verbose', 'warm_start', 'n_jobs'],
        'relevantToOptimizer': ['penalty', 'dual', 'tol', 'fit_intercept', 'intercept_scaling', 'solver', 'max_iter', 'multi_class'],
        'additionalProperties': False,
        'properties': {
            'penalty': {
                'XXX TODO XXX': "str, 'l1' or 'l2', default: 'l2'",
                'description': 'Used to specify the norm used in the penalization',
                'enum': ['l2'],
                'default': 'l2'},
            'dual': {
                'type': 'boolean',
                'default': False,
                'description': 'Dual or primal formulation'},
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
                'description': 'Inverse of regularization strength; must be a positive float'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Specifies if a constant (a.k.a'},
            'intercept_scaling': {
                'type': 'number',
                'minimumForOptimizer': 0.0,
                'maximumForOptimizer': 1.0,
                'distribution': 'uniform',
                'default': 1,
                'description': "Useful only when the solver 'liblinear' is used and self.fit_intercept is set to True"},
            'class_weight': {
                'XXX TODO XXX': "dict or 'balanced', default: None",
                'description': 'Weights associated with classes in the form ``{class_label: weight}``',
                'enum': ['balanced'],
                'default': 'balanced'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The seed of the pseudo random number generator to use when shuffling the data'},
            'solver': {
                'enum': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                'default': 'liblinear',
                'description': 'Algorithm to use in the optimization problem'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 100,
                'description': 'Useful only for the newton-cg, sag and lbfgs solvers'},
            'multi_class': {
                'enum': ['auto', 'liblinear', 'multinomial', 'ovr'],
                'default': 'ovr',
                'description': "If the option chosen is 'ovr', then a binary problem is fit for each label"},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.'},
            'warm_start': {
                'type': 'boolean',
                'default': False,
                'description': 'When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of CPU cores used when parallelizing over classes if multi_class=\'ovr\'"'},
        }}, {
        'XXX TODO XXX': 'Parameter: penalty > only l2 penalties'}, {
        'XXX TODO XXX': 'Parameter: dual > only implemented for l2 penalty with liblinear solver'}, {
        'XXX TODO XXX': "Parameter: intercept_scaling > only when the solver 'liblinear' is used and self"}, {
        'XXX TODO XXX': "Parameter: solver > only 'newton-cg'"}, {
        'description': 'max_iter, only for the newton-cg',
        'anyOf': [{
            'type': 'object',
            'properties': {
                'max_iter': {
                    'enum': [100]},
            }}, {
            'type': 'object',
            'properties': {
                'newton-cg': {
                    'enum': ['the']},
            }}]}],
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
            'description': 'Target vector relative to X.'},
        'sample_weight': {
            'laleType': 'Any',
            'XXX TODO XXX': 'array-like, shape (n_samples,) optional',
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
_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Probability estimates.',
    'type': 'object',
    'required': ['X'],
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
    'description': 'Returns the probability of the sample for each class in the model, where classes are ordered as they are in ``self.classes_``.',
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
LogisticRegression = lale.operators.make_operator(LogisticRegressionImpl, _combined_schemas)

