from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from models import Classifier

# Model checkpoints, training logs, and training summaries
TRAINING_DIR_NAME = 'training'
TRAIN_EVAL_DIR_NAME = 'train_eval'  # Training evaluation summaries
VALID_EVAL_DIR_NAME = 'valid_eval'  # Validation evaluation summaries

# Graph collection keys for summaries we want to restrict to training or
# evaluation
EVAL_SUMMARIES = ['EVAL_SUMMARIES']
TRAINING_SUMMARIES = ['TRAINING_SUMMARIES']


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
        assert(labels.dtype in [tf.int32, tf.int64]
               )  # labels are always integers
        with tf.variable_scope(name):
            mask = tf.where(tf.equal(labels, -1), tf.zeros_like(labels, dtype=dtype),
                            tf.ones_like(labels, dtype=dtype))
        return mask

    def get_weights(self, inputs):
        with tf.variable_scope('weights'):
            labels = inputs['labels']
            ambig_mask = self.get_ambiguous_mask(labels)
            # TODO(cprobert): we can support additional weights here
            weights = ambig_mask
        return weights

    def get_preds_and_binary_preds(self, logits):
        with tf.variable_scope('preds'):
            preds = tf.sigmoid(logits, name='sigmoid')
            binary_preds = tf.greater(preds, 0.5)
        return preds, binary_preds

    def get_binary_labels(self, labels):
        with tf.variable_scope('binary_labels'):
            binary_labels = tf.equal(labels, 1)
        return binary_labels

    def metrics_by_task(self, logits, labels, weights, task_names, prefix='train-by-TF'):
        with tf.variable_scope('metrics_by_task'):
            tasknames_to_metrics = {}
            for i, task_name in enumerate(task_names):
                t_logits = logits[:, i]
                t_labels = labels[:, i]
                t_weights = weights[:, i]
                t_prefix = '{}-{}'.format(prefix, task_name)
                tasknames_to_metrics[task_name] = self.get_merged_metrics(
                    t_logits, t_labels, t_weights, prefix=t_prefix)
            for task_name, (names_to_values, _) in tasknames_to_metrics.items():
                for metric_name, metric_value in names_to_values.items():
                    tf.summary.scalar(metric_name, metric_value)

    def metrics_by_dataset(self, logits, labels, weights, dataset_idxs, dataset_names,
                           prefix='train-by-celltype'):
        with tf.variable_scope('metrics_by_dataset'):
            datasetnames_to_metrics = {}
            for i, dataset_name in enumerate(dataset_names):
                d_mask = tf.where(tf.equal(dataset_idxs, i),
                                  tf.zeros_like(labels, dtype=tf.float32),
                                  tf.ones_like(labels, dtype=tf.float32))
                d_weights = tf.multiply(d_mask, weights)
                d_prefix = '{}-{}'.format(prefix, dataset_name)
                datasetnames_to_metrics[dataset_name] = self.get_merged_metrics(
                    logits, labels, d_weights, prefix=d_prefix)
            for dataset_name, (names_to_values, _) in datasetnames_to_metrics.items():
                for metric_name, metric_value in names_to_values.items():
                    tf.summary.scalar(metric_name, metric_value)

    def get_merged_metrics(self, logits, labels, weights, prefix='train-eval'):
        """Returns `names_to_values` and `names_to_updates` dicts."""
        preds, binary_preds = self.get_preds_and_binary_preds(logits)
        binary_labels = self.get_binary_labels(labels)

        names_to_metrics = {
            '{}/Accuracy'.format(prefix): tf.contrib.metrics.streaming_accuracy(
                binary_preds, binary_labels, weights=weights),
            '{}/Recall'.format(prefix): tf.contrib.metrics.streaming_recall(
                preds, labels, weights=weights),
            '{}/Precision'.format(prefix): tf.contrib.metrics.streaming_precision(
                preds, labels, weights=weights),
            '{}/auROC'.format(prefix): tf.contrib.metrics.streaming_auc(
                preds, labels, weights=weights, curve='ROC', name='auROC'),
            '{}/auPRC'.format(prefix): tf.contrib.metrics.streaming_auc(
                preds, labels, weights=weights, curve='PR', name='auPRC'),
        }

        for specificity in [0.01, 0.05, 0.1, 0.25]:
            name = '{0}/sensty_at_{1:.2f}_specfty'.format(prefix, specificity)
            names_to_metrics[name] = tf.contrib.metrics.streaming_sensitivity_at_specificity(
                preds, labels, specificity, weights=weights, name=name)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
            names_to_metrics)
        return names_to_values, names_to_updates

    def get_summaries_for_metrics(self, names_to_values):
        summary_ops = []
        for metric_name, metric_value in names_to_values.iteritems():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)
        return summary_ops

    def get_loss(self, logits, labels, weights):
        sigmoid_xentropy_loss = slim.losses.sigmoid_cross_entropy(
            logits, labels, weights=weights)
        tf.scalar_summary('simoid-xentropy-loss', sigmoid_xentropy_loss)

        # this adds regularization if it's specified
        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('total-loss', total_loss)

        return total_loss

    def get_logits_labels_loss_weights(self, examples_queue):
        inputs = examples_queue.outputs
        labels = inputs["labels"]

        logits = self.model.get_logits(inputs)
        weights = self.get_weights(inputs)
        loss = self.get_loss(logits, labels, weights)

        return logits, labels, loss, weights

    def train(self, examples_queue, log_dir):
        logits, labels, loss, weights = self.get_logits_labels_loss_weights(
            examples_queue)
        task_names = examples_queue.task_names
        dataset_names = examples_queue.dataset_labels
        dataset_idxs = examples_queue.outputs['dataset-index']

        # TODO(cprobert): maybe move this to a "calculate_metrics" function?
        # TODO(cprobert): possibly store metrics in graph collections, which would let us
        # decide which ones to calculate at training / evaluation time?
        names_to_values, names_to_updates = self.get_merged_metrics(
            logits, labels, weights, prefix='train-eval')
        metrics_summary_ops = self.get_summaries_for_metrics(names_to_values)
        self.metrics_by_task(logits, labels, weights, task_names)
        self.metrics_by_dataset(logits, labels, weights,
                                dataset_idxs, dataset_names)

        opt = self.optimizer(self.lr)
        train_op = slim.learning.create_train_op(
            loss, opt, clip_gradient_norm=2.0, summarize_gradients=True,
            colocate_gradients_with_ops=True)

        total_steps = int(self.num_epochs * self.epoch_size / self.batch_size)

        # note: we can add other summaries here
        summary_op = tf.summary.merge(metrics_summary_ops)
        # but aren't all our summaries already under graphkeys.summaries?

        # TODO(cprobert): implement early stopping?
        slim.learning.train(
            train_op, log_dir, number_of_steps=total_steps, save_summaries_secs=10,
            trace_every_n_steps=1000)
