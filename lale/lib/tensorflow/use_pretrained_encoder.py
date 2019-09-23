import os
import shutil
import logging
import random
from lale.operators import make_operator
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import lale.helpers

logging.basicConfig(level=logging.INFO)


class UsePretrainedEncoderImpl(object):
    def __init__(self,
                 url='/tmp/tfhub/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47',
                 batch_size=32):

        if os.path.exists(url):
            self.url = url
        else:
            os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub'
            self.url = "https://tfhub.dev/google/universal-sentence-encoder/2"

        # load the use model from saved location or tensorflow hub
        self.embed = hub.Module(self.url, trainable=True)
        self.sess = tf.Session()
        self.epochs = 10
        self.lr = 0.01
        self.batch_size = batch_size
        self.sess.run([tf.global_variables_initializer(),
                       tf.tables_initializer()])

    def fit(self, X, Y, model_dir='./fine_tuned_use'):
        """
        method for fine-tune the universal sentence encoder using text classification task;
        fine-tune the current model and save the fine_tuned model for later application
        Parameters
        ----------
        X : list of strings, input corpus for fine-tune use
        Y : list of integers, input label for fine-tune use
        model_dir: directory to save the fine_tuned model for later use
        Returns
        -------
        """
        Y = np.array(Y).reshape(-1)
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
                if current_step+batch_size < len(x):
                    x_batch = x[current_step:current_step+batch_size]
                    y_batch = y[current_step:current_step+batch_size]
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
        iterations = int(np.ceil(len(X)/self.batch_size))
        self.sess.run(tf.global_variables_initializer())
        for _ in range(self.epochs):
            for _ in range(iterations):
                X, Y, x_batch, y_batch, curr = next_batch(X, Y, curr)
                self.sess.run(train_opt, feed_dict={X_batch: x_batch, Y_batch: y_batch})

        # save model
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        self.embed.export(model_dir, session=self.sess)

    def transform(self, x):
        """
        method for encoding strings into floating point arrays using universal sentence encoder
        Parameters
        ----------
        X : list of strings, input corpus for fine-tune use
        Y : list of integers, input label for fine-tune use
        model_dir: directory to save the fine_tuned model for later use
        Returns
        -------
        """
        sentence_embedding = self.embed(x)
        transformed_x = self.sess.run(sentence_embedding)
        return transformed_x


_input_schema_fit = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for training.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Features',
            'anyOf': [
                {'type': 'array',
                 'items': {'type': 'string'}},
                {'type': 'array',
                 'items': {
                     'type': 'array', 'minItems': 1, 'maxItems': 1,
                     'items': {'type': 'string'}}}]},
        'y': {
            'description': 'Labels, required',
            'anyOf': [
                {'type': 'array',
                 'items': {'type': 'number'}}
            ]
        }
    }}

_output_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Output data schema for transformed data.',
    'type': 'array',
    'items': {'type': 'array', 'items': {'type': 'number'}}}

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Hyperparameter schema.',
    'allOf': [
        {'description':
             'This first sub-object lists all constructor arguments with their '
             'types, one at a time, omitting cross-argument constraints.',
         'type': 'object',
         'additionalProperties': False,
         'relevantToOptimizer': [],
         'properties': {
             'batch_size': {
                 'description': 'Batch size used for transform.',
                 'type': 'integer',
                 'default': 32,
                 'minimum': 1,
                 'distribution': 'uniform',
                 'minimumForOptimizer': 16,
                 'maximumForOptimizer': 128}}}]}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters for a transformer for'
                   ' a text data transformer based on pre-trained USE model '
                   '(https://tfhub.dev/google/universal-sentence-encoder/2).',
    'type': 'object',
    'tags': {
        'pre': ['text'],
        'op': ['transformer', '~interpretable'],
        'post': ['embedding']},
    'properties': {
        'input_fit': _input_schema_fit,
        'output': _output_schema,
        'hyperparams': _hyperparams_schema}}

if __name__ == "__main__":
    lale.helpers.validate_is_schema(_combined_schemas)

UsePretrainedEncoder = make_operator(UsePretrainedEncoderImpl, _combined_schemas)
