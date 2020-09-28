from numpy import inf, nan
from sklearn.kernel_approximation import AdditiveChi2Sampler as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class AdditiveChi2SamplerImpl:
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
            "relevantToOptimizer": ["sample_steps"],
            "additionalProperties": False,
            "properties": {
                "sample_steps": {
                    "type": "integer",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 3,
                    "distribution": "uniform",
                    "default": 2,
                    "description": "Gives the number of (complex) sampling points.",
                },
                "sample_interval": {
                    "anyOf": [{"type": "number"}, {"enum": [None]}],
                    "default": None,
                    "description": "Sampling interval",
                },
            },
        }
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
set_docstrings(AdditiveChi2SamplerImpl, _combined_schemas)
AdditiveChi2Sampler = make_operator(AdditiveChi2SamplerImpl, _combined_schemas)
