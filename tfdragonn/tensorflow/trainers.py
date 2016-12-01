from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tfdragonn.tensorflow import Classifier

class ClassiferTrainer(object):

    def __init__(self, model, optimizer=tf.train.AdamOptimizer, lr=0.0003,
                 early_stopping_metric='auPRC', num_epochs=100,
                 batch_size=32, epoch_size=250000, early_stopping_patience=5,
                 save_best_model_to_prefix=None):
        assert isinstance(model, Classifier)

        self.model = model
        self.optimizer= optimizer
        self.lr = lr
        self.early_stopping_metric =  early_stopping_metric
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.early_stopping_patience = early_stopping_patience
        self.save_best_model_to_prefix = save_best_model_to_prefix

    def get_loss(self, logits, labels):
        task_losses = slim.losses.sigmoid_cross_entropy(predictions, labels)
        total_loss = slim.losses.get_total_loss()
        return total_loss

    def test_in_session(self, sess, dataset_id2data_queue):
        """
        Returns
        -------
        dataset2classification_result : dict
        combined_classification_result : instance of ClassificationResult
        """
        pass

    def train_in_session(self, sess, dataset_id2train_data_queue, dataset_id2test_data_queue):
        sess.run(tf.initialize_local_variables())
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess)
        # TODO create master training queue
        # create training loop
        inputs_batch = training_queue.dequeue_many(self.batch_size)
        logits = model.get_logits(inputs_batch)
        loss = self.get_loss(logits, inputs_batch["labels"])
        train_op = slim.learning.create_train_op(loss, self.optimizer(self.lr))
        batches_per_epoch = int(self.epoch_zie / self.batch_size)
        valid_metrics = []
        best_metric = 0
        for epoch in range(1, self.num_epochs + 1):
            for batch_indx in xrange(1, batches_per_epoch + 1):
                sess.run(train_op)
            # get metrics on validation data
            dataset_id2metrics, total_metrics = self.test_in_session(sess, dataset_id2test_data_queue)
            valid_metrics.append(epoch_valid_metrics)
            # early stop or continue
            current_metric = total_metrics[early_stopping_metric].mean()
            if (early_stopping_metric == 'Loss') == (current_metric <= best_metric):
                best_metric = current_metric
                best_epoch = epoch
                early_stopping_wait = 0
                if self.save_best_model_to_prefix is not None:
                    self.save(self.save_best_model_to_prefix)
            else:
                if early_stopping_wait >= early_stopping_patience:
                    break
                early_stopping_wait += 1
