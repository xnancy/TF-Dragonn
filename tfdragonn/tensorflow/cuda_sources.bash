#!/usr/bin/env bash

# Sources required by TensorFlow 0.12 (includes libcupti)

export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:/usr/local/cudnn-5.1/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64/:${LD_LIBRARY_PATH}"
