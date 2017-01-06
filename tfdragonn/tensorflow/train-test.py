
# coding: utf-8

# In[1]:

import os.path

import tensorflow as tf
from tensorflow.contrib import slim

import dataset_interval_reader
from shared_examples_queue import SharedExamplesQueue
from models import SequenceAndDnaseClassifier
from trainers import ClassiferTrainer


# In[2]:

NUM_TASKS = 1

DATA_DIR = 'test-data'
LOG_DIR = 'test-logs-3'

# INTERVALS_FILE = os.path.join(DATA_DIR, 'intervals_file.json')
# INPUTS_FILE = os.path.join(DATA_DIR, 'inputs_file.json')

INPUTS_FILE = "/users/jisraeli/src/tf-dragonn_tensorflow/tfdragonn/tensorflow/examples/processed_inputs_example.json"
INTERVALS_FILE = "/users/jisraeli/src/tf-dragonn_tensorflow/tfdragonn/tensorflow/examples/myc_conservative_dnase_regions_and_labels_stride200_flank400.json"


# In[3]:

get_ipython().system(u' rm -rf {LOG_DIR}/*')


# In[4]:

readers, task_names = dataset_interval_reader.get_readers_and_tasks(INPUTS_FILE, INTERVALS_FILE)
shared_queue = SharedExamplesQueue(readers, task_names, batch_size=128)


# In[5]:

model = SequenceAndDnaseClassifier(num_tasks=NUM_TASKS)


# In[6]:

trainer = ClassiferTrainer(model, epoch_size=5000)


# In[ ]:

trainer.train(shared_queue, LOG_DIR)
