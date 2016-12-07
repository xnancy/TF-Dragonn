from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from models import Classifier

TRAIN_DIR_NAME = 'train'
VALID_DIR_NAME = 'valid'
TEST_DIR_NAME = 'test'


class ClassiferTrainer(object):

    def __init__(self, model, optimizer=tf.train.AdamOptimizer, lr=0.0003,
                 early_stopping_metric='auPRC', num_epochs=100, batch_size=32, epoch_size=250000,
                 early_stopping_patience=5, save_best_model_to_prefix=None):
        assert isinstance(model, Classifier)

        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.early_stopping_metric = early_stopping_metric
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.early_stopping_patience = early_stopping_patience
        self.save_best_model_to_prefix = save_best_model_to_prefix

    def get_ambiguous_mask(self, labels, dtype=tf.float32, name='ambiguous-examples-mask'):
        """Return a weights matrix with the same size as labels. Entries in labels
            equal to -1 are masked with `0`, all other values are `1`.
        """
        assert(labels.dtype in [tf.int32, tf.int64])
        with tf.variable_scope(name):
            mask = tf.where(tf.equal(labels, -1), tf.zeros_like(labels, dtype=dtype),
                            tf.ones_like(labels, dtype=dtype))
        return mask

    def get_loss(self, logits, labels):
        ambig_mask = self.get_ambiguous_mask(labels)
        weights = ambig_mask  # TODO(cprobert): we could support additional weights here

        sigmoid_xentropy_loss = slim.losses.sigmoid_cross_entropy(logits, labels, weights=weights)
        tf.scalar_summary(sigmoid_xentropy_loss, 'simoid-xentropy-loss')

        total_loss = slim.losses.get_total_loss()  # this adds regularization if it's specified
        tf.scalar_summary(total_loss, 'total-loss')

        return total_loss

    def test_in_session(self, sess, dataset_id2data_queue):
        """
        Returns
        -------
        dataset2classification_result : dict
        combined_classification_result : instance of ClassificationResult
        """
        pass

    def train(self, shared_examples_queue, log_dir):
        inputs = shared_examples_queue.outputs
        labels = inputs["labels"]

        logits = self.model.get_logits(inputs)
        loss = self.get_loss(logits, labels)
        opt = self.optimizer(self.lr)

        train_op = slim.learning.create_train_op(
            loss, opt, clip_gradient_norm=4.0, summarize_gradients=True)

        total_steps = int(self.num_epochs * self.epoch_size / self.batch_size)

        # Todo: implement early stopping
        slim.learning.train(
            train_op, log_dir, number_of_steps=total_steps, save_summaries_secs=10,
            trace_every_n_steps=100)
