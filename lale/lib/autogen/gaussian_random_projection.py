from numpy import inf, nan
from sklearn.random_projection import GaussianRandomProjection as Op

from lale.docstrings import set_docstrings
from lale.operators import make_operator


class _GaussianRandomProjectionImpl:
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
    "description": "inherited docstring for GaussianRandomProjection    Reduce dimensionality through Gaussian random projection",
    "allOf": [
        {
            "type": "object",
            "required": ["n_components", "eps", "random_state"],
            "relevantToOptimizer": ["n_components", "eps"],
            "additionalProperties": False,
            "properties": {
                "n_components": {
                    "XXX TODO XXX": "int or 'auto', optional (default = 'auto')",
                    "description": "Dimensionality of the target projection space",
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimumForOptimizer": 2,
                            "maximumForOptimizer": 256,
                            "distribution": "uniform",
                        },
                        {"enum": ["auto"]},
                    ],
                    "default": "auto",
                },
                "eps": {
                    "XXX TODO XXX": "strictly positive float, optional (default=0.1)",
                    "description": "Parameter to control the quality of the embedding according to the Johnson-Lindenstrauss lemma when n_components is set to 'auto'",
                    "type": "number",
                    "minimumForOptimizer": 0.001,
                    "maximumForOptimizer": 0.1,
                    "distribution": "loguniform",
                    "default": 0.1,
                },
                "random_state": {
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "numpy.random.RandomState"},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Control the pseudo random number generator used to generate the matrix at fit time",
                },
            },
        },
        {
            "description": "eps=%f and n_samples=%d lead to a target dimension of %d which is larger than the original space with n_features=%d'      % (self.eps, n_samples, self.n_components_, n_features) ",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"n_components": {"not": {"enum": ["auto"]}}},
                },
                {
                    "XXX TODO XXX": "johnson_lindenstrauss_min_dim(n_samples=X.shape[0], eps=self.eps) <= 0"
                },
                {
                    "XXX TODO XXX": "johnson_lindenstrauss_min_dim(n_samples=X.shape[0], eps=self.eps) <= X.shape[1]"
                },
            ],
        },
    ],
}
_input_fit_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Generate a sparse random projection matrix",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "numpy array or scipy.sparse of shape [n_samples, n_features]",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "Training set: only the shape is used to find optimal random matrix dimensions based on the theory referenced in the afore mentioned papers.",
        },
        "y": {"laleType": "Any", "XXX TODO XXX": "", "description": "Ignored"},
    },
}
_input_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Project the data by using matrix product with the random matrix",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
                    "XXX TODO XXX": "numpy array or scipy.sparse of shape [n_samples, n_features]",
                },
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            ],
            "description": "The input data to project into a smaller dimensional space.",
        }
    },
}
_output_transform_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Projected array.",
    "anyOf": [
        {
            "type": "array",
            "items": {"laleType": "Any", "XXX TODO XXX": "item type"},
            "XXX TODO XXX": "numpy array or scipy sparse of shape [n_samples, n_components]",
        },
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    ],
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "documentation_url": "https://scikit-learn.org/0.20/modules/generated/sklearn.random_projection.GaussianRandomProjection#sklearn-random_projection-gaussianrandomprojection",
    "import_from": "sklearn.random_projection",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}
GaussianRandomProjection = make_operator(
    _GaussianRandomProjectionImpl, _combined_schemas
)

set_docstrings(GaussianRandomProjection)
