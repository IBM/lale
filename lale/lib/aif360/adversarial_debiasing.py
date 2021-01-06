# Copyright 2020 IBM Corporation
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

import aif360.algorithms.inprocessing

try:
    import tensorflow as tf

    tensorflow_installed = True
except ImportError:
    tensorflow_installed = False

import lale.docstrings
import lale.operators

from .util import (
    _BaseInprocessingImpl,
    _categorical_fairness_properties,
    _categorical_input_predict_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
)


class AdversarialDebiasingImpl(_BaseInprocessingImpl):
    def __init__(
        self,
        favorable_labels,
        protected_attributes,
        preprocessing=None,
        scope_name="adversarial_debiasing",
        sess=None,
        seed=None,
        adversary_loss_weight=0.1,
        num_epochs=50,
        batch_size=128,
        classifier_num_hidden_units=200,
        debias=True,
    ):
        assert tensorflow_installed, """Your Python environment does not have tensorflow installed. You can install it with
    pip install tensorflow
or with
    pip install 'lale[full]'"""
        assert "1.13.1" <= tf.__version__ <= "2", tf.__version__
        if sess is None:
            sess = tf.Session()
        prot_attr_names = [pa["feature"] for pa in protected_attributes]
        unprivileged_groups = [{name: 0 for name in prot_attr_names}]
        privileged_groups = [{name: 1 for name in prot_attr_names}]
        mitigator = aif360.algorithms.inprocessing.AdversarialDebiasing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            scope_name=scope_name,
            sess=sess,
            seed=seed,
            adversary_loss_weight=adversary_loss_weight,
            num_epochs=num_epochs,
            batch_size=batch_size,
            classifier_num_hidden_units=classifier_num_hidden_units,
            debias=debias,
        )
        super(AdversarialDebiasingImpl, self).__init__(
            favorable_labels=favorable_labels,
            protected_attributes=protected_attributes,
            preprocessing=preprocessing,
            mitigator=mitigator,
        )


_input_fit_schema = _categorical_supervised_input_fit_schema
_input_predict_schema = _categorical_input_predict_schema
_output_predict_schema = _categorical_output_predict_schema

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                *_categorical_fairness_properties.keys(),
                "preprocessing",
                "scope_name",
                "sess",
                "seed",
                "adversary_loss_weight",
                "num_epochs",
                "batch_size",
                "classifier_num_hidden_units",
                "debias",
            ],
            "relevantToOptimizer": [
                "adversary_loss_weight",
                "num_epochs",
                "batch_size",
                "classifier_num_hidden_units",
            ],
            "properties": {
                **_categorical_fairness_properties,
                "preprocessing": {
                    "description": "Transformer, which may be an individual operator or a sub-pipeline.",
                    "anyOf": [
                        {"laleType": "operator"},
                        {"description": "lale.lib.lale.NoOp", "enum": [None]},
                    ],
                    "default": None,
                },
                "scope_name": {
                    "description": "Scope name for the tenforflow variables.",
                    "type": "string",
                    "default": "adversarial_debiasing",
                },
                "sess": {
                    "description": "TensorFlow session.",
                    "anyOf": [
                        {
                            "description": "User-provided session object.",
                            "laleType": "Any",
                        },
                        {
                            "description": "Create a session for the user.",
                            "enum": [None],
                        },
                    ],
                    "default": None,
                },
                "seed": {
                    "description": "Seed to make `predict` repeatable.",
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                },
                "adversary_loss_weight": {
                    "description": "Hyperparameter that chooses the strength of the adversarial loss.",
                    "type": "number",
                    "default": 0.1,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
                "num_epochs": {
                    "description": "Number of training epochs.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 50,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 500,
                },
                "batch_size": {
                    "description": "Batch size.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 128,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 4,
                    "maximumForOptimizer": 512,
                },
                "classifier_num_hidden_units": {
                    "description": "Number of hidden units in the classifier model.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 200,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 16,
                    "maximumForOptimizer": 1024,
                },
                "debias": {
                    "description": "Learn a classifier with or without debiasing.",
                    "type": "boolean",
                    "default": True,
                },
            },
        },
    ],
}

_combined_schemas = {
    "description": """`AdversarialDebiasing`_ in-processing operator for fairness mitigation.

.. _`AdversarialDebiasing`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.AdversarialDebiasing.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.prejudice_remover.html",
    "import_from": "aif360.sklearn.inprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

lale.docstrings.set_docstrings(AdversarialDebiasingImpl, _combined_schemas)

AdversarialDebiasing = lale.operators.make_operator(
    AdversarialDebiasingImpl, _combined_schemas
)
