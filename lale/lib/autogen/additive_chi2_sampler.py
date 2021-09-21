from numpy import inf, nan
from sklearn.kernel_approximation import AdditiveChi2Sampler as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _AdditiveChi2SamplerImpl:
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


_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "inherited docstring for AdditiveChi2Sampler    Approximate feature map for additive chi2 kernel.",
    "allOf": [
        {
            "type": "object",
            "required": ["sample_steps", "sample_interval"],
            "relevantToOptimizer": ["sample_steps", "sample_interval"],
            "additionalProperties": False,
            "properties": {
                "sample_steps": {
                    "type": "integer",
                    "minimumForOptimizer": 1,
                    "maximumForOptimizer": 5,
                    "distribution": "uniform",
                    "default": 2,
                    "description": "Gives the number of (complex) sampling points.",
                },
                "sample_interval": {
                    "anyOf": [
                        {
                            "type": "number",
                            "minimumForOptimizer": 0.1,
                            "maximumForOptimizer": 1.0,
                            "distribution": "uniform",
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Sampling interval",
                },
            },
        },
        {
            "description": "From /kernel_approximation.py:AdditiveChi2Sampler:fit, Exception: raise ValueError(     'If "
            "sample_steps is not in [1, 2, 3], you need to provide sample_interval') ",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"sample_interval": {"not": {"enum": [None]}}},
                },
                {"type": "object", "properties": {"sample_steps": {"enum": [1, 2, 3]}}},
            ],
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Set the parameters",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training data, where n_samples in the number of samples and n_features is the number of features.",
        }
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Apply approximate feature map to X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Whether the return value is an array of sparse matrix depends on the type of the input X.",
    "laleType": "Any",
    "XXX TODO XXX": "{array, sparse matrix},                shape = (n_samples, n_features * (2*sample_steps + 1))",
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.kernel_approximation.AdditiveChi2Sampler#sklearn-kernel_approximation-additivechi2sampler",
    "import_from": "sklearn.kernel_approximation",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
AdditiveChi2Sampler = make_operator(_AdditiveChi2SamplerImpl, _combined_schemas)

set_docstrings(AdditiveChi2Sampler)
