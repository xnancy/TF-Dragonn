from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def classification_metrics(logits, labels, weights, dataset_idxs, dataset_names, task_names):
    with tf.variable_scope('metrics'):
        create_all_metrics(logits, labels, weights, 'metrics')
    metrics_by_task(logits, labels, weights, task_names)
    metrics_by_dataset(logits, labels, weights, dataset_idxs, dataset_names)


def metrics_by_task(self, logits, labels, weights, task_names, prefix='TF'):
    with tf.variable_scope('metrics_by_task'):
        for i, task_name in enumerate(task_names):
            t_logits = logits[:, i]
            t_labels = labels[:, i]
            t_weights = weights[:, i]
            t_prefix = '{}-{}'.format(prefix, task_name)
            create_all_metrics(t_logits, t_labels, t_weights, t_prefix)


def metrics_by_dataset(self, logits, labels, weights, dataset_idxs, dataset_names,
                       prefix='celltype'):
    with tf.variable_scope('metrics_by_dataset'):
        for i, dataset_name in enumerate(dataset_names):
            d_mask = tf.where(tf.equal(dataset_idxs, i),
                              tf.ones_like(labels, dtype=tf.float32),
                              tf.zeros_like(labels, dtype=tf.float32))
            d_weights = tf.multiply(d_mask, weights)
            d_prefix = '{}-{}'.format(prefix, dataset_name)
            create_all_metrics(logits, labels, d_weights, d_prefix)


def create_all_metrics(logits, labels, weights, prefix):
    """Create summaries for all metrics for a given set of logits/labels/weights."""
    names_to_values, names_to_updates = get_merged_classification_metrics(
        logits, labels, weights, prefix)
    other_names_to_updates = get_additional_metrics(logits, labels, weights, prefix)
    for metric_key, metric_update in other_names_to_updates.items():
        names_to_updates[metric_key] = metric_update
        names_to_values[metric_key] = metric_update
    register_summaries_for_metrics(names_to_updates)
    return names_to_values, names_to_updates


def get_merged_classification_metrics(logits, labels, weights, prefix):
    """Returns `names_to_values` and `names_to_updates` dicts for tf-slim metrics."""
    preds = tf.sigmoid(logits, name='sigmoid')
    binary_preds = tf.greater(preds, 0.5)
    binary_labels = tf.equal(labels, 1)

    # Weirdly, it looks like Recall/Precision expect bools encoded as ints
    binary_preds_ints = tf.cast(binary_preds, tf.int32)
    binary_labels_ints = tf.cast(binary_labels, tf.int32)

    names_to_metrics = {
        '{}/Accuracy'.format(prefix): tf.contrib.metrics.streaming_accuracy(
            binary_preds, binary_labels, weights=weights),
        '{}/Recall'.format(prefix): tf.contrib.metrics.streaming_recall(
            binary_preds_ints, binary_labels_ints, weights=weights),
        '{}/Precision'.format(prefix): tf.contrib.metrics.streaming_precision(
            binary_preds_ints, binary_labels_ints, weights=weights),
        '{}/auROC'.format(prefix): tf.contrib.metrics.streaming_auc(
            preds, labels, weights=weights, curve='ROC', name='auROC'),
        '{}/auPRC'.format(prefix): tf.contrib.metrics.streaming_auc(
            preds, labels, weights=weights, curve='PR', name='auPRC'),
        '{}/label-balance'.format(prefix): tf.contrib.metrics.streaming_mean(
            labels, weights=weights, name='label-balance'),
    }

    for specificity in [0.01, 0.05, 0.1, 0.25]:
        name = '{0}/sensitivity_at_{1:.2f}_specfty'.format(prefix, specificity)
        names_to_metrics[name] = tf.contrib.metrics.streaming_sensitivity_at_specificity(
            preds, labels, specificity, weights=weights, name=name)

    # TODO(cprobert): add recall-at-K metrics
    # TODO(cprobert): add recall-at-FDR metrics

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(names_to_metrics)

    return names_to_values, names_to_updates


def get_additional_metrics(logits, labels, weights, prefix):
    """Some metrics aren't avalible as tf-slim metrics."""
    names_to_updates = {}

    loss = slim.losses.sigmoid_cross_entropy(logits, labels, weights=weights)
    loss_name = '{}/xentropy-loss'.format(prefix)
    names_to_updates[loss_name] = loss

    return names_to_updates


def register_summaries_for_metrics(names_to_updates):
    for metric_name, metric_update in names_to_updates.items():
        tf.summary.scalar(metric_name, metric_update, description=metric_name)
