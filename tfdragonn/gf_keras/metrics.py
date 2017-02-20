from __future__ import absolute_import, division, print_function
import numpy as np
from collections import OrderedDict
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

AMBIG_LABEL = -1

def positive_accuracy(labels, predictions, threshold=0.5):
    return 100 * (predictions[labels == 1] > threshold).mean()


def negative_accuracy(labels, predictions, threshold=0.5):
    return 100 * (predictions[labels == 0] < threshold).mean()


def balanced_accuracy(labels, predictions, threshold=0.5):
    return (positive_accuracy(labels, predictions, threshold) +
            negative_accuracy(labels, predictions, threshold)) / 2


def auROC(labels, predictions):
    return roc_auc_score(labels, predictions)


def auPRC(labels, predictions):
    precision, recall = precision_recall_curve(labels, predictions)[:2]
    return auc(recall, precision)


def recall_at_precision_threshold(labels, predictions, precision_threshold):
    precision, recall = precision_recall_curve(labels, predictions)[:2]
    return 100 * recall[np.searchsorted(precision - precision_threshold, 0)]


class ClassificationResult(object):

    def __init__(self, labels, predictions, task_names=None):
        assert labels.dtype == int
        self.results = []
        non_ambig_task_names = []
        for i, (task_labels, task_predictions) in enumerate(zip(labels.T, predictions.T)):
            non_ambig_label_indxs = np.where(task_labels != AMBIG_LABEL)[0]
            if len(non_ambig_label_indxs) == 0: # skip ambiguous tasks
                continue
            task_labels = task_labels[non_ambig_label_indxs]
            task_predictions = task_predictions[non_ambig_label_indxs]
            if task_names is not None:
                non_ambig_task_names.append(task_names[i])
            self.results.append(OrderedDict((
                ('Balanced accuracy', balanced_accuracy(
                    task_labels, task_predictions)),
                ('auROC', auROC(task_labels, task_predictions)),
                ('auPRC', auPRC(task_labels, task_predictions)),
                ('Recall at 5% FDR', recall_at_precision_threshold(
                    task_labels, task_predictions, 0.95)),
                ('Recall at 10% FDR', recall_at_precision_threshold(
                    task_labels, task_predictions, 0.9)),
                ('Recall at 25% FDR', recall_at_precision_threshold(
                    task_labels, task_predictions, 0.75)),
                ('Recall at 50% FDR', recall_at_precision_threshold(
                    task_labels, task_predictions, 0.5)),
                ('Num Positives', task_labels.sum()),
                ('Num Negatives', (1 - task_labels).sum())
            )))
        self.task_names = task_names if task_names is None else non_ambig_task_names
        self.multitask = labels.shape[1] > 1

    def __str__(self):
        return '\n'.join(
            '{}Balanced Accuracy: {:.2f}%\t'
            'auROC: {:.3f}\t auPRC: {:.3f}\n'
            'Recall at 5% | 10% | 25% | 50% FDR: {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}%\t'
            'Num Positives: {}\t Num Negatives: {}\n'.format(
                '{}: '.format('Task {}'.format(
                    self.task_names[task_index]
                    if self.task_names is not None else task_index))
                if self.multitask else '', *results.values())
            for task_index, results in enumerate(self.results))

    def __getitem__(self, item):
        return np.array([task_results[item] for task_results in self.results])
