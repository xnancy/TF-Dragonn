from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

from metrics import classification_metrics
from models import Classifier


class ClassiferTrainer(object):

    def __init__(self, optimizer=tf.train.AdamOptimizer, lr=0.0003, epoch_size=250000):

        self.optimizer = optimizer
        self.lr = lr
        self.epoch_size = epoch_size
        self._current_step_limit = 0

    def get_ambiguous_mask(self, labels, dtype=tf.float32, name='ambiguous-examples-mask'):
        """Return a weights matrix with the same size as labels. Entries in labels
            equal to -1 are masked with `0`, all other values are `1`.
        """
        assert(labels.dtype in [tf.int32, tf.int64])  # labels are always integers
        with tf.variable_scope(name):
            mask = tf.where(tf.equal(labels, -1), tf.zeros_like(labels, dtype=dtype),
                            tf.ones_like(labels, dtype=dtype))
        return mask

    def get_weights(self, inputs):
        with tf.variable_scope('weights'):
            labels = inputs['labels']
            ambig_mask = self.get_ambiguous_mask(labels)
            # NB: we can support additional weights here if needed
            weights = ambig_mask
        return weights

    def get_loss(self, logits, labels, weights):
        with tf.variable_scope('loss'):
            sigmoid_xentropy_loss = slim.losses.sigmoid_cross_entropy(logits, labels, weights=weights)
            tf.summary.scalar('simoid-xentropy-loss', sigmoid_xentropy_loss)

            # this adds regularization if it's specified
            total_loss = slim.losses.get_total_loss()
            tf.summary.scalar('total-loss', total_loss)

            return total_loss

    def get_logits_labels_loss_weights(self, model, examples_queue):
        inputs = examples_queue.outputs
        labels = inputs["labels"]

        logits = model.get_logits(inputs)
        weights = self.get_weights(inputs)
        loss = self.get_loss(logits, labels, weights)

        return logits, labels, loss, weights

    def train(self, model, examples_queue, train_log_dir, checkpoint=None,
              session_config=None, num_epochs=1):
        logits, labels, loss, weights = self.get_logits_labels_loss_weights(model, examples_queue)
        task_names = examples_queue.task_names
        dataset_names = examples_queue.dataset_labels
        dataset_idxs = examples_queue.outputs['dataset-index']

        names_to_values, names_to_updates = classification_metrics(
            logits, labels, weights, dataset_idxs, dataset_names, task_names)

        opt = self.optimizer(self.lr)
        train_op = slim.learning.create_train_op(
            loss, opt, clip_gradient_norm=2.0, summarize_gradients=True,
            colocate_gradients_with_ops=True, update_ops=names_to_updates.values())

        if checkpoint is not None:
            variables = slim.get_model_variables()
            init_op, init_feeddict = slim.assign_from_checkpoint(checkpoint, variables)

            def InitAssignFn(sess):
                sess.run(init_op, init_feeddict)
        else:
            InitAssignFn = None

        batch_size = dataset_idxs.get_shape()[0].value
        self._current_step_limit += int(num_epochs * self.epoch_size / batch_size)

        slim.learning.train(
            train_op, train_log_dir, number_of_steps=self._current_step_limit,
            save_summaries_secs=10, trace_every_n_steps=100000, save_interval_secs=120,
            session_config=session_config, graph=tf.get_default_graph(), init_fn=InitAssignFn)

        checkpoint_regex = os.path.join(train_log_dir, 'model.ckpt*index')
        checkpoint_fname = max(glob.iglob(
            checkpoint_regex), key=os.path.getctime)
        checkpoint_fname = checkpoint_fname.rstrip('.index')
        return checkpoint_fname

    def evaluate(self, model, examples_queue, num_evals, valid_log_dir, checkpoint, session_config=None):
        logits, labels, loss, weights = self.get_logits_labels_loss_weights(model, examples_queue)
        task_names = examples_queue.task_names
        dataset_names = examples_queue.dataset_labels
        dataset_idxs = examples_queue.outputs['dataset-index']

        names_to_values, names_to_updates = classification_metrics(
            logits, labels, weights, dataset_idxs, dataset_names, task_names)

        eval_results = slim.evaluation.evaluate_once(
            master='', checkpoint_path=checkpoint, logdir=valid_log_dir, num_evals=num_evals,
            eval_op=names_to_updates.values(), session_config=session_config,
            final_op=names_to_values)

        print(eval_results)
        return eval_results
