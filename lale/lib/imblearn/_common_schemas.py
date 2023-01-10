# Copyright 2023 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict

JSON_TYPE = Dict[str, Any]


_hparam_kind_sel = {
    "description": """Strategy to use in order to exclude samples.
If ``all``, all neighbours will have to agree with the samples of interest to not be excluded.
If ``mode``, the majority vote of the neighbours will be used in order to exclude a sample.""",
    "enum": ["all", "mode"],
    "default": "all",
}

_hparam_n_jobs = {
    "description": "The number of threads to open if possible.",
    "type": "integer",
    "default": 1,
}

_hparam_n_neighbors: JSON_TYPE = {
    "description": "Number of neighbors.",
    "anyOf": [
        {
            "type": "integer",
            "description": "Number of nearest neighbours to used to construct synthetic samples.",
        },
        {
            "laleType": "Any",
            "description": "An estimator that inherits from :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to find the `n_neighbors`.",
        },
    ],
}

_hparam_operator = {
    "description": """Trainable Lale pipeline that is trained using the data obtained from the current imbalance corrector.

Predict, transform, predict_proba or decision_function would just be
forwarded to the trained pipeline.  If operator is a Planned pipeline,
the current imbalance corrector can't be trained without using an
optimizer to choose a trainable operator first. Please refer to
lale/examples for more examples.""",
    "laleType": "operator",
}

_hparam_random_state = {
    "description": "Control the randomization of the algorithm.",
    "anyOf": [
        {
            "description": "RandomState used by np.random",
            "enum": [None],
        },
        {
            "description": "The seed used by the random number generator",
            "type": "integer",
        },
        {
            "description": "Random number generator instance.",
            "laleType": "numpy.random.RandomState",
        },
    ],
    "default": None,
}

_hparam_sampling_strategy_number = {
    "type": "number",
    "forOptimizer": False,
    "description": """Desired ratio of the number of samples in the
minority class over the number of samples in the majority class after
resampling. Therefore, the ratio is expressed as :math:`\\alpha_{os} =
N_{rm} / N_{M}` where :math:`N_{rm}` is the number of samples in the
minority class after resampling and :math:`N_{M}` is the number of
samples in the majority class.

.. warning::
    Only available for **binary** classification.
    An error is raised for multi-class classification.""",
}

_hparam_sampling_strategy_enum = {
    "enum": ["minority", "not minority", "not majority", "all", "auto"],
    "description": """The class targeted by the resampling.
The number of samples in the different classes will be equalized.
Possible choices are:

- ``'minority'``: resample only the minority class;
- ``'not minority'``: resample all classes but the minority class;
- ``'not majority'``: resample all classes but the majority class;
- ``'all'``: resample all classes;
- ``'auto'``: equivalent to ``'not majority'``.""",
}

_hparam_sampling_strategy_object = {
    "type": "object",
    "forOptimizer": False,
    "description": "Keys correspond to the targeted classes and values correspond to the desired number of samples for each targeted class.",
}

_hparam_sampling_strategy_callable = {
    "laleType": "callable",
    "forOptimizer": False,
    "description": """Function taking ``y`` and returns a ``dict``.
The keys correspond to the targeted classes and the values correspond to the desired number of samples for each class.""",
}

_hparam_sampling_strategy_list = {
    "description": "Classes targeted by the resampling.",
    "forOptimizer": False,
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
    ],
}

_hparam_sampling_strategy_anyof_elc = {
    "description": "Sampling information to resample the data set.",
    "anyOf": [
        _hparam_sampling_strategy_enum,
        _hparam_sampling_strategy_list,
        _hparam_sampling_strategy_callable,
    ],
    "default": "auto",
}

_hparam_sampling_strategy_anyof_neoc = {
    "description": "Sampling information to resample the data set.",
    "anyOf": [
        _hparam_sampling_strategy_number,
        _hparam_sampling_strategy_enum,
        _hparam_sampling_strategy_object,
        _hparam_sampling_strategy_callable,
    ],
    "default": "auto",
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
    },
}

_input_fit_schema_cats = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"enum": [None]},
            ],
        },
    },
}

_input_transform_schema_cats = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"enum": [None]},
            ],
        },
    },
}

_output_transform_schema = {
    "description": "Output data schema for transformed data.",
    "laleType": "Any",
}

_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_input_predict_schema_cats = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_output_predict_schema = {
    "description": "Output data schema for predictions.",
    "laleType": "Any",
}

_output_predict_proba_schema = {
    "description": "Probability of the sample for each class in the model.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_output_decision_function_schema = {
    "description": "Output data schema for predictions.",
    "laleType": "Any",
}
