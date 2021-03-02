from numpy import inf, nan
from sklearn.gaussian_process import GaussianProcessClassifier as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _GaussianProcessClassifierImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for GaussianProcessClassifier    Gaussian process classification (GPC) based on Laplace approximation.",
    "allOf": [
        {
            "type": "object",
            "required": [
                "kernel",
                "optimizer",
                "n_restarts_optimizer",
                "max_iter_predict",
                "warm_start",
                "copy_X_train",
                "random_state",
                "multi_class",
                "n_jobs",
            ],
            "relevantToOptimizer": [
                "optimizer",
                "n_restarts_optimizer",
                "max_iter_predict",
                "multi_class",
            ],
            "additionalProperties": False,
            "properties": {
                "kernel": {
                    "XXX TODO XXX": "kernel object",
                    "description": "The kernel specifying the covariance function of the GP",
                    "enum": [None],
                    "default": None,
                },
                "optimizer": {
                    "anyOf": [
                        {"laleType": "callable", "forOptimizer": False},
                        {"enum": ["fmin_l_bfgs_b"]},
                    ],
                    "default": "fmin_l_bfgs_b",
                    "description": "Can either be one of the internally supported optimizers for optimizing the kernel's parameters, specified by a string, or an externally defined optimizer passed as a callable",
                },
                "n_restarts_optimizer": {
                    "type": "integer",
                    "minimumForOptimizer": 0,
                    "maximumForOptimizer": 1,
                    "distribution": "uniform",
                    "default": 0,
                    "description": "The number of restarts of the optimizer for finding the kernel's parameters which maximize the log-marginal likelihood",
                },
                "max_iter_predict": {
                    "type": "integer",
                    "minimumForOptimizer": 100,
                    "maximumForOptimizer": 101,
                    "distribution": "uniform",
                    "default": 100,
                    "description": "The maximum number of iterations in Newton's method for approximating the posterior during predict",
                },
                "warm_start": {
                    "type": "boolean",
                    "default": False,
                    "description": "If warm-starts are enabled, the solution of the last Newton iteration on the Laplace approximation of the posterior mode is used as initialization for the next call of _posterior_mode()",
                },
                "copy_X_train": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, a persistent copy of the training data is stored in the object",
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "The generator used to initialize the centers",
                },
                "multi_class": {
                    "XXX TODO XXX": "string, default",
                    "description": "Specifies how multi-class classification problems are handled",
                    "enum": [
                        "auto",
                        "crammer_singer",
                        "liblinear",
                        "multinomial",
                        "one_vs_one",
                        "one_vs_rest",
                        "ovr",
                    ],
                    "default": "one_vs_rest",
                },
                "n_jobs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": 1,
                    "description": "The number of jobs to use for the computation",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit Gaussian process classification model",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target values, must be binary",
        },
    },
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Perform classification on an array of test vectors X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predicted target values for X, values are from ``classes_``",
    "type": "array",
    "items": {"type": "number"},
}
_input_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Return probability estimates for the test vector X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Returns the probability of the samples for each class in the model",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier#sklearn-gaussian_process-gaussianprocessclassifier",
    "import_from": "sklearn.gaussian_process",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}
GaussianProcessClassifier = make_operator(
    _GaussianProcessClassifierImpl, _combined_schemas
)

set_docstrings(GaussianProcessClassifier)
