import sklearn
from numpy import inf, nan
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _LinearDiscriminantAnalysisImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)

    def decision_function(self, X):
        return self._wrapped_model.decision_function(X)


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for LinearDiscriminantAnalysis    Linear Discriminant Analysis",
    "allOf": [
        {
            "type": "object",
            "required": [
                "solver",
                "shrinkage",
                "priors",
                "n_components",
                "store_covariance",
                "tol",
            ],
            "relevantToOptimizer": ["solver", "n_components", "tol", "shrinkage"],
            "additionalProperties": False,
            "properties": {
                "solver": {
                    "enum": ["eigen", "lsqr", "svd"],
                    "default": "svd",
                    "description": "Solver to use, possible values:   - 'svd': Singular value decomposition (default)",
                },
                "shrinkage": {
                    "anyOf": [
                        {"enum": ["auto"]},
                        {
                            "type": "number",
                            "minimumForOptimizer": 0,
                            "maximumForOptimizer": 1,
                            "minimum": 0,
                            "maximum": 1,
                            "exclusiveMinimum": True,
                            "exclusiveMaximum": True,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Shrinkage parameter, possible values:   - None: no shrinkage (default)",
                },
                "priors": {
                    "XXX TODO XXX": "array, optional, shape (n_classes,)",
                    "description": "Class priors.",
                    "enum": [None],
                    "default": None,
                },
                "n_components": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimun": 1,
                            "laleMaximum": "X/items/maxItems",
                            "minimumForOptimizer": 2,
                            "maximumForOptimizer": 256,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Number of components (< n_classes - 1) for dimensionality reduction.",
                },
                "store_covariance": {
                    "type": "boolean",
                    "default": False,
                    "description": "Additionally compute class covariance matrix (default False), used only in 'svd' solver",
                },
                "tol": {
                    "type": "number",
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                    "distribution": "loguniform",
                    "default": 0.0001,
                    "description": "Threshold used for rank estimation in SVD solver",
                },
            },
        },
        {
            "description": "shrinkage, only with 'lsqr' and 'eigen' solvers",
            "anyOf": [
                {"type": "object", "properties": {"shrinkage": {"enum": [None]}}},
                {
                    "type": "object",
                    "properties": {"solver": {"enum": ["lsqr", "eigen"]}},
                },
            ],
        },
        {
            "description": "store_covariance, only in 'svd' solver",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"store_covariance": {"enum": [False]}},
                },
                {"type": "object", "properties": {"solver": {"enum": ["svd"]}}},
            ],
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Fit LinearDiscriminantAnalysis model according to the given",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data.",
        },
        "y": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Target values.",
        },
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Project data to maximize class separation.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Input data.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Transformed data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_input_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict class labels for samples in X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array_like or sparse matrix, shape (n_samples, n_features)",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Samples.",
        }
    },
}
_output_predict_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predicted class label per sample.",
    "type": "array",
    "items": {"type": "number"},
}
_input_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Estimate probability.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Input data.",
        }
    },
}
_output_predict_proba_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Estimated probabilities.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_input_decision_function_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Predict confidence scores for samples.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "array_like or sparse matrix, shape (n_samples, n_features)",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Samples.",
        }
    },
}
_output_decision_function_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Confidence scores per (sample, class) combination",
    "laleType": "Any",
    "XXX TODO XXX": "array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis#sklearn-discriminant_analysis-lineardiscriminantanalysis",
    "import_from": "sklearn.discriminant_analysis",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer", "estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}
LinearDiscriminantAnalysis = make_operator(
    _LinearDiscriminantAnalysisImpl, _combined_schemas
)

if sklearn.__version__ >= "0.24":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis#sklearn-discriminant_analysis-lineardiscriminantanalysis
    # new: https://scikit-learn.org/0.24/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis#sklearn-discriminant_analysis-lineardiscriminantanalysis
    LinearDiscriminantAnalysis = LinearDiscriminantAnalysis.customize_schema(
        covariance_estimator={
            "anyOf": [
                {
                    "type": "string",
                    "forOptimizer": False,
                },
                {"enum": [None]},
            ],
            "default": None,
            "description": "type of (covariance estimator). Estimate the covariance matrices instead of relying on the empirical covariance estimator (with potential shrinkage)",
        },
        set_as_available=True,
    )
    LinearDiscriminantAnalysis = LinearDiscriminantAnalysis.customize_schema(
        constraint={
            "description": "covariance estimator is not supported with svd solver. Try another solver",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"solver": {"not": {"enum": ["svd"]}}},
                },
                {
                    "type": "object",
                    "properties": {"covariance_estimator": {"enum": [None]}},
                },
            ],
        },
        set_as_available=True,
    )
    LinearDiscriminantAnalysis = LinearDiscriminantAnalysis.customize_schema(
        constraint={
            "description": "covariance_estimator and shrinkage parameters are not None. Only one of the two can be set.",
            "anyOf": [
                {"type": "object", "properties": {"solver": {"enum": ["svd", "lsqr"]}}},
                {
                    "type": "object",
                    "properties": {"solver": {"not": {"enum": ["eigen"]}}},
                },
                {
                    "type": "object",
                    "properties": {"covariance_estimator": {"enum": [None]}},
                },
                {"type": "object", "properties": {"shrinkage": {"enum": [None, 0]}}},
            ],
        },
        set_as_available=True,
    )

set_docstrings(LinearDiscriminantAnalysis)
