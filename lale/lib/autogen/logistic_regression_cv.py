
from sklearn.linear_model.logistic import LogisticRegressionCV as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LogisticRegressionCVImpl():

    def __init__(self, Cs=10, fit_intercept=True, cv=3, dual=False, penalty='l2', scoring=None, solver='lbfgs', tol=0.0001, max_iter=100, class_weight='balanced', n_jobs=None, verbose=0, refit=True, intercept_scaling=1.0, multi_class='ovr', random_state=None):
        self._hyperparams = {
            'Cs': Cs,
            'fit_intercept': fit_intercept,
            'cv': cv,
            'dual': dual,
            'penalty': penalty,
            'scoring': scoring,
            'solver': solver,
            'tol': tol,
            'max_iter': max_iter,
            'class_weight': class_weight,
            'n_jobs': n_jobs,
            'verbose': verbose,
            'refit': refit,
            'intercept_scaling': intercept_scaling,
            'multi_class': multi_class,
            'random_state': random_state}

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
    'description': 'inherited docstring for LogisticRegressionCV    Logistic Regression CV (aka logit, MaxEnt) classifier.',
    'allOf': [{
        'type': 'object',
        'relevantToOptimizer': ['Cs', 'fit_intercept', 'cv', 'dual', 'penalty', 'scoring', 'solver', 'tol', 'max_iter', 'multi_class'],
        'additionalProperties': False,
        'properties': {
            'Cs': {
                'XXX TODO XXX': 'list of floats | int',
                'description': 'Each of the values in Cs describes the inverse of regularization',
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 11,
                'distribution': 'uniform',
                'default': 10},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Specifies if a constant (a.k.a. bias or intercept) should be'},
            'cv': {
                'XXX TODO XXX': 'integer or cross-validation generator, default: None',
                'description': 'The default cross-validation generator used is Stratified K-Folds.',
                'type': 'integer',
                'minimumForOptimizer': 3,
                'maximumForOptimizer': 4,
                'distribution': 'uniform',
                'default': 3},
            'dual': {
                'type': 'boolean',
                'default': False,
                'description': 'Dual or primal formulation. Dual formulation is only implemented for'},
            'penalty': {
                'XXX TODO XXX': "str, 'l1' or 'l2'",
                'description': "Used to specify the norm used in the penalization. The 'newton-cg',",
                'enum': ['l2'],
                'default': 'l2'},
            'scoring': {
                'anyOf': [{
                    'type': 'object',
                    'forOptimizer': False}, {
                    'enum': ['accuracy', None]}],
                'default': None,
                'description': 'A string (see model evaluation documentation) or'},
            'solver': {
                'enum': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                'default': 'lbfgs',
                'description': 'Algorithm to use in the optimization problem.'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'Tolerance for stopping criteria.'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 100,
                'description': 'Maximum number of iterations of the optimization algorithm.'},
            'class_weight': {
                'XXX TODO XXX': "dict or 'balanced', optional",
                'description': 'Weights associated with classes in the form ``{class_label: weight}``.',
                'enum': ['balanced'],
                'default': 'balanced'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of CPU cores used during the cross-validation loop.'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': "For the 'liblinear', 'sag' and 'lbfgs' solvers set verbose to any"},
            'refit': {
                'type': 'boolean',
                'default': True,
                'description': 'If set to True, the scores are averaged across all folds, and the'},
            'intercept_scaling': {
                'type': 'number',
                'default': 1.0,
                'description': "Useful only when the solver 'liblinear' is used"},
            'multi_class': {
                'enum': ['auto', 'liblinear', 'multinomial', 'ovr'],
                'default': 'ovr',
                'description': "If the option chosen is 'ovr', then a binary problem is fit for each"},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator;'},
        }}, {
        'description': 'dual, XXX TODO XXX, only implemented for l2 penalty with liblinear solver'}, {
        'description': 'penalty, XXX TODO XXX, only l2 penalties'}, {
        'description': "solver, XXX TODO XXX, only 'newton-cg'"}, {
        'description': "intercept_scaling, XXX TODO XXX, only when the solver 'liblinear' is used and self"}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model according to the given training data.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Training vector, where n_samples is the number of samples and'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target vector relative to X.'},
        'sample_weight': {
            'XXX TODO XXX': 'array-like, shape (n_samples,) optional',
            'description': 'Array of weights that are assigned to individual samples.'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class labels for samples in X.',
    'type': 'object',
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
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
    'description': 'Returns the probability of the sample for each class in the model,',
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
LogisticRegressionCV = lale.operators.make_operator(LogisticRegressionCVImpl, _combined_schemas)

