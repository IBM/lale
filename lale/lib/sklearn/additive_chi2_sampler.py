from sklearn.kernel_approximation import AdditiveChi2Sampler as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator

_hyperparams_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Approximate feature map for additive chi2 kernel.",
    "allOf": [
        {
            "type": "object",
            "required": ["sample_steps", "sample_interval"],
            "relevantToOptimizer": ["sample_steps"],
            "additionalProperties": False,
            "properties": {
                "sample_steps": {
                    "type": "integer",
                    "minimumForOptimizer": 1,
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
        },
        {
            "description": "sample_interval must be specified when sample_steps not in {1,2,3}",
            "anyOf": [
                {
                    "type": "object",
                    "properties:": {"sample_steps": {"enum": [1, 2, 3]}},
                },
                {
                    "type": "object",
                    "properties:": {"sample_interval": {"type": "number"}},
                },
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
    "description": """`Approximate feature map for additive chi2 kernel`_ from sklearn

.. _`Approximate feature map for additive chi2 kernel.`: https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.AdditiveChi2Sampler
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.additive_chi2_sampler.html",
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
AdditiveChi2Sampler = make_operator(Op, _combined_schemas)

set_docstrings(AdditiveChi2Sampler)
