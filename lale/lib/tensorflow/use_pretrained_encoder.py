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

import os
import random
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import lale.docstrings
import lale.operators


class _USEPretrainedEncoderImpl(object):
    def __init__(self, model_path=None, batch_size=32):
        self.resources_dir = os.path.join(os.path.dirname(__file__), "resources")

        if model_path is None:
            model_path = os.path.join(
                "pretrained_USE", "1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47"
            )
        if os.path.exists(os.path.join(self.resources_dir, model_path)):
            self.url = os.path.join(self.resources_dir, model_path)
        else:
            os.environ["TFHUB_CACHE_DIR"] = os.path.join(
                self.resources_dir, "pretrained_USE"
            )
            self.url = "https://tfhub.dev/google/universal-sentence-encoder/2"

        # load the use model from saved location or tensorflow hub
        self.embed = hub.Module(self.url, trainable=True)
        self.sess = tf.Session()
        self.epochs = 10
        self.lr = 0.01
        self.batch_size = batch_size
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def fit(self, X, y, model_dir=None):
        Y = np.array(y).reshape(-1)
        num_classes = len(np.unique(Y))
        Y = np.eye(num_classes)[Y]

        def next_batch(x, y, current_step, batch_size=self.batch_size):
            """method for extracton of next batch of data"""
            if len(x) < batch_size:
                data_idx = np.arange((len(x))).tolist()
                random.shuffle(data_idx)
                x_batch = [x[idx] for idx in data_idx]
                y_batch = [y[idx] for idx in data_idx]
            else:
                if current_step + batch_size < len(x):
                    x_batch = x[current_step : current_step + batch_size]
                    y_batch = y[current_step : current_step + batch_size]
                    current_step += batch_size
                else:
                    x_batch = x[current_step:]
                    y_batch = y[current_step:]

                    data_idx = np.arange((len(x))).tolist()
                    random.shuffle(data_idx)
                    x = [x[idx] for idx in data_idx]
                    y = [y[idx] for idx in data_idx]
                    current_step = 0

            return x, y, x_batch, y_batch, current_step

        X_batch = tf.placeholder(dtype=tf.string, shape=(None))
        Y_batch = tf.placeholder(dtype=tf.int16, shape=(None, num_classes))
        embedded_representation = self.embed(X_batch)
        dense_layer = tf.layers.Dense(units=num_classes, use_bias=True)
        logits = dense_layer(embedded_representation)
        los = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_batch, logits=logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_opt = optimizer.minimize(loss=los)

        curr = 0
        iterations = int(np.ceil(len(X) / self.batch_size))
        self.sess.run(tf.global_variables_initializer())
        for _ in range(self.epochs):
            for _ in range(iterations):
                X, Y, x_batch, y_batch, curr = next_batch(X, Y, curr)
                self.sess.run(train_opt, feed_dict={X_batch: x_batch, Y_batch: y_batch})

        # save model
        if model_dir is None:
            model_dir = os.path.join(self.resources_dir, "fine_tuned_USE")

        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        self.embed.export(model_dir, session=self.sess)

        return self

    def transform(self, X):
        sentence_embedding = self.embed(X)
        transformed_x = self.sess.run(sentence_embedding)
        return transformed_x


_input_schema_fit = {
    "description": "Input data schema for training.",
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Input corpus for fine-tuning.",
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"type": "string"},
                    },
                },
            ],
        },
        "y": {
            "description": "Input labels for fine-tuning",
            "type": "array",
            "items": {"type": "number"},
        },
        "model_dir": {
            "description": "Directory to save the fine_tuned model for later use.",
            "type": "string",
        },
    },
}

_input_transform_schema = {
    "description": "Input data schema for training.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Input Text",
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"type": "string"},
                    },
                },
            ],
        }
    },
}

_output_transform_schema = {
    "description": "Output data schema for transformed data.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {
                "batch_size": {
                    "description": "Batch size used for transform.",
                    "type": "integer",
                    "default": 32,
                    "minimum": 1,
                    "distribution": "uniform",
                    "minimumForOptimizer": 16,
                    "maximumForOptimizer": 128,
                }
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """USE (Universal Sentence Encoder) text embedding (https://tfhub.dev/google/universal-sentence-encoder/2).

References
----------
Daniel Cer, et al. Universal Sentence Encoder. arXiv:1803.11175, 2018
""",
    "type": "object",
    "tags": {
        "pre": ["text"],
        "op": ["transformer", "~interpretable"],
        "post": ["embedding"],
    },
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_schema_fit,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


USEPretrainedEncoder = lale.operators.make_operator(
    _USEPretrainedEncoderImpl, _combined_schemas
)

lale.docstrings.set_docstrings(USEPretrainedEncoder)
