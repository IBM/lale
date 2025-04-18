# Copyright 2020-2023 IBM Corporation
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

import contextlib
import io
import os
import uuid

import packaging.version

import lale.docstrings
import lale.operators

# suppress spurious warnings from TensorFlow that are caused by
# indirectly importing it via aif360.algorithms.inprocessing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import aif360.algorithms.inprocessing  # noqa:E402 # pylint:disable=wrong-import-position,wrong-import-order

from .util import (  # noqa:E402 # pylint:disable=wrong-import-position,wrong-import-order
    _BaseInEstimatorImpl,
    _categorical_fairness_properties,
    _categorical_input_predict_proba_schema,
    _categorical_input_predict_schema,
    _categorical_output_predict_proba_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
)

try:
    import tensorflow as tf
except ImportError:
    tf = None


class _AdversarialDebiasingImpl(_BaseInEstimatorImpl):
    def __init__(  # pylint:disable=super-init-not-called
        self,
        *,
        favorable_labels,
        protected_attributes,
        unfavorable_labels=None,
        redact=True,
        preparation=None,
        scope_name="adversarial_debiasing",
        verbose=0,
        **hyperparams,
    ):
        assert (
            tf is not None
        ), """Your Python environment does not have tensorflow installed. You can install it with
    pip install tensorflow
or with
    pip install 'lale[full]'"""
        tf_version = packaging.version.parse(getattr(tf, "__version__"))
        assert packaging.version.Version("1.13.1") <= tf_version, tf_version
        self.scope_name = scope_name
        self.protected_attributes = protected_attributes
        self.favorable_labels = favorable_labels
        self.unfavorable_labels = unfavorable_labels
        self.redact = redact
        self.preparation = preparation
        self.verbose = verbose
        self.hyperparams = hyperparams

    def fit(self, X, y=None):
        assert tf is not None
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        if self.hyperparams.get("sess", None) is None:
            self.hyperparams["sess"] = tf.compat.v1.Session()
        prot_attr_names = [pa["feature"] for pa in self.protected_attributes]
        unprivileged_groups = [{name: 0 for name in prot_attr_names}]
        privileged_groups = [{name: 1 for name in prot_attr_names}]
        mitigator = aif360.algorithms.inprocessing.AdversarialDebiasing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            scope_name=self.scope_name + str(uuid.uuid4()),
            **self.hyperparams,
        )
        super().__init__(
            favorable_labels=self.favorable_labels,
            protected_attributes=self.protected_attributes,
            unfavorable_labels=self.unfavorable_labels,
            redact=self.redact,
            preparation=self.preparation,
            mitigator=mitigator,
        )
        if self.verbose == 0:
            with contextlib.redirect_stdout(io.StringIO()):
                super().fit(X, y)
        else:
            super().fit(X, y)
        return self


_input_fit_schema = _categorical_supervised_input_fit_schema
_input_predict_schema = _categorical_input_predict_schema
_output_predict_schema = _categorical_output_predict_schema
_input_predict_proba_schema = _categorical_input_predict_proba_schema
_output_predict_proba_schema = _categorical_output_predict_proba_schema

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                *_categorical_fairness_properties.keys(),
                "redact",
                "preparation",
                "scope_name",
                "sess",
                "seed",
                "adversary_loss_weight",
                "num_epochs",
                "batch_size",
                "classifier_num_hidden_units",
                "debias",
                "verbose",
            ],
            "relevantToOptimizer": [
                "adversary_loss_weight",
                "num_epochs",
                "batch_size",
                "classifier_num_hidden_units",
            ],
            "properties": {
                **_categorical_fairness_properties,
                "redact": {
                    "description": "Whether to redact protected attributes before data preparation (recommended) or not.",
                    "type": "boolean",
                    "default": True,
                },
                "preparation": {
                    "description": "Transformer, which may be an individual operator or a sub-pipeline.",
                    "anyOf": [
                        {"laleType": "operator"},
                        {"description": "lale.lib.lale.NoOp", "enum": [None]},
                    ],
                    "default": None,
                },
                "scope_name": {
                    "description": "Scope name for the tenforflow variables. A unique alpha-numeric suffix is added to this value.",
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
                "verbose": {
                    "description": "If zero, then no output.",
                    "type": "integer",
                    "default": 0,
                },
            },
        },
    ],
}

_combined_schemas = {
    "description": """`AdversarialDebiasing`_ in-estimator fairness mitigator. Learns a classifier to maximize prediction accuracy and simultaneously reduce an adversary's ability to determine the protected attribute from the predictions (`Zhang et al. 2018`_). This approach leads to a fair classifier as the predictions cannot carry any group discrimination information that the adversary can exploit. Implemented based on TensorFlow.

.. _`AdversarialDebiasing`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.AdversarialDebiasing.html
.. _`Zhang et al. 2018`: https://doi.org/10.1145/3278721.3278779
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.adversarial_debiasing.html#lale.lib.aif360.adversarial_debiasing.AdversarialDebiasing",
    "import_from": "aif360.sklearn.inprocessing",
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

AdversarialDebiasing = lale.operators.make_operator(
    _AdversarialDebiasingImpl, _combined_schemas
)

lale.docstrings.set_docstrings(AdversarialDebiasing)
