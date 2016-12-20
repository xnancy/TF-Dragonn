FROM kundajelab/cuda-anaconda-base:latest

MAINTAINER Chris Probert <cprobert@stanford.edu>

#
# Setup and activate the anaconda environment
#
COPY tfbind.yml .
RUN conda env create -f tfbind.yml
ENV PATH /opt/conda/envs/tfbind/bin:$PATH

#
# Install tensorflow
#
RUN pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc1-cp27-none-linux_x86_64.whl
