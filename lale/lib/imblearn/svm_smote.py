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

from imblearn.over_sampling import SVMSMOTE as OrigModel

import lale.docstrings
import lale.operators
from lale.lib.imblearn.base_resampler import (
    BaseResamplerImpl,
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


class SVMSMOTEImpl(BaseResamplerImpl):
    def __init__(
        self,
        operator=None,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=1,
        m_neighbors=10,
        svm_estimator=None,
        out_step=0.5,
    ):
        if operator is None:
            raise ValueError("Operator is a required argument.")

        self._hyperparams = {
            "sampling_strategy": sampling_strategy,
            "random_state": random_state,
            "k_neighbors": k_neighbors,
            "n_jobs": n_jobs,
            "m_neighbors": m_neighbors,
            "svm_estimator": svm_estimator,
            "out_step": out_step,
        }

        resampler_instance = OrigModel(**self._hyperparams)
        super(SVMSMOTEImpl, self).__init__(
            operator=operator, resampler=resampler_instance
        )


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": ["operator"],
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
The values correspond to the desired number of samples for each targeted
class.""",
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
                "k_neighbors": {
                    "description": """If ``int``, number of nearest neighbours to used to construct synthetic samples.
If object, an estimator that inherits from
:class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
find the k_neighbors.""",
                    "anyOf": [{"laleType": "Any"}, {"type": "integer"}],
                    "default": 5,
                },
                "n_jobs": {
                    "description": "The number of threads to open if possible.",
                    "type": "integer",
                    "default": 1,
                },
                "m_neighbors": {
                    "description": """If ``int``, number of nearest neighbours to use to determine if a minority sample is in danger.
If object, an estimator that inherits from
:class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to find the m_neighbors.""",
                    "anyOf": [{"laleType": "Any"}, {"type": "integer"}],
                    "default": 10,
                },
                "svm_estimator": {
                    "description": "A parametrized sklearn.svm.SVC classifier can be passed.",
                    "anyOf": [{"laleType": "Any"}, {"enum": [None]}],
                    "default": None,
                },
                "out_step": {
                    "description": "Step size when extrapolating.",
                    "type": "number",
                    "default": 0.5,
                },
            },
        }
    ]
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Over-sampling using SVM-SMOTE,
Variant of SMOTE algorithm which use an SVM algorithm to detect sample to use for generating new synthetic samples.""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.imblearn.svm_smote.html",
    "import_from": "imblearn.over_sampling",
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

lale.docstrings.set_docstrings(SVMSMOTEImpl, _combined_schemas)

SVMSMOTE = lale.operators.make_operator(SVMSMOTEImpl, _combined_schemas)
