# Copyright 2019 IBM Corporation
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

from imblearn.under_sampling import InstanceHardnessThreshold as OrigModel

import lale.docstrings
import lale.operators
from lale.lib.imblearn.base_resampler import (
    _BaseResamplerImpl,
    _input_decision_function_schema,
    _input_fit_schema,
    _input_predict_proba_schema,
    _input_predict_schema,
    _input_transform_schema,
    _output_decision_function_schema,
    _output_predict_proba_schema,
    _output_predict_schema,
    _output_transform_schema,
)


class _InstanceHardnessThresholdImpl(_BaseResamplerImpl):
    def __init__(
        self,
        operator=None,
        estimator=None,
        sampling_strategy="auto",
        random_state=None,
        cv=5,
        n_jobs=1,
    ):
        if operator is None:
            raise ValueError("Operator is a required argument.")

        self._hyperparams = {
            "estimator": estimator,
            "sampling_strategy": sampling_strategy,
            "random_state": random_state,
            "cv": cv,
            "n_jobs": n_jobs,
        }

        resampler_instance = OrigModel(**self._hyperparams)
        super(_InstanceHardnessThresholdImpl, self).__init__(
            operator=operator, resampler=resampler_instance
        )


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "relevantToOptimizer": ["operator"],
            "additionalProperties": False,
            "properties": {
                "operator": {
                    "description": """Trainable Lale pipeline that is trained using the data obtained from the current imbalance corrector.
Predict, transform, predict_proba or decision_function would just be forwarded to the trained pipeline.
If operator is a Planned pipeline, the current imbalance corrector can't be trained without using an optimizer to
choose a trainable operator first. Please refer to lale/examples for more examples.""",
                    "anyOf": [{"laleType": "operator"}],
                },
                "estimator": {
                    "description": """Classifier to be used to estimate instance hardness of the samples.
By default a :class:`sklearn.ensemble.RandomForestClassifer` will be used.
If ``str``, the choices using a string are the following: ``'knn'``,
``'decision-tree'``, ``'random-forest'``, ``'adaboost'``,
``'gradient-boosting'`` and ``'linear-svm'``.  If object, an estimator
inherited from :class:`sklearn.base.ClassifierMixin` and having an
attribute :func:`predict_proba`.""",
                    "anyOf": [
                        {"laleType": "Any"},
                        {
                            "enum": [
                                "knn",
                                "decision-tree",
                                "random-forest",
                                "adaboost",
                                "gradient-boosting",
                                "linear-svm",
                            ]
                        },
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "sampling_strategy": {
                    "description": """sampling_strategy : float, str, dict or callable, default='auto'.
Sampling information to resample the data set.
""",
                    "anyOf": [
                        {
                            "description": """When ``float``,
it corresponds to the desired ratio of the number of
samples in the minority class over the number of samples in the
majority class after resampling. Therefore, the ratio is expressed as
:math:`\\alpha_{os} = N_{rm} / N_{M}` where :math:`N_{rm}` is the
number of samples in the minority class after resampling and
:math:`N_{M}` is the number of samples in the majority class.
.. warning::
    ``float`` is only available for **binary** classification. An
    error is raised for multi-class classification.""",
                            "type": "number",
                        },
                        {
                            "description": """When ``str``, specify the class targeted by the resampling.
The number of samples in the different classes will be equalized.
Possible choices are:
``'minority'``: resample only the minority class;
``'not minority'``: resample all classes but the minority class;
``'not majority'``: resample all classes but the majority class;
``'all'``: resample all classes;
``'auto'``: equivalent to ``'not majority'``.""",
                            "enum": [
                                "minority",
                                "not minority",
                                "not majority",
                                "all",
                                "auto",
                            ],
                        },
                        {
                            "description": """- When ``dict``, the keys correspond to the targeted classes.
The values correspond to the desired number of samples for each targeted class.""",
                            "type": "object",
                        },
                        {
                            "description": """When callable, function taking ``y`` and returns a ``dict``.
The keys correspond to the targeted classes. The values correspond to the
desired number of samples for each class.""",
                            "laleType": "callable",
                        },
                    ],
                    "default": "auto",
                },
                "random_state": {
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
                },
                "cv": {
                    "description": "Number of folds to be used when estimating samples’ instance hardness.",
                    "type": "integer",
                    "default": 5,
                },
                "n_jobs": {
                    "description": "The number of threads to open if possible.",
                    "type": "integer",
                    "default": 1,
                },
            },
        }
    ]
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Class to perform under-sampling based on the instance hardness threshold.""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.imblearn.instance_hardness_threshold.html",
    "import_from": "imblearn.under_sampling",
    "type": "object",
    "tags": {
        "pre": [],
        "op": [
            "transformer",
            "estimator",
            "resampler",
        ],  # transformer and estimator both as a higher-order operator
        "post": [],
    },
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


InstanceHardnessThreshold = lale.operators.make_operator(
    _InstanceHardnessThresholdImpl, _combined_schemas
)

lale.docstrings.set_docstrings(InstanceHardnessThreshold)
