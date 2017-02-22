FROM kundajelab/genomeflow:latest

MAINTAINER Chris Probert <cprobert@stanford.edu>

#
# Install tfdragonn dependencies
#
COPY conda_requirements.txt .
RUN conda install -y --file  conda_requirements.txt -c astro -c bioconda -c r -c defaults -c conda-forge && conda clean -t

COPY pip_requirements.txt .
RUN pip install --no-cache-dir -r pip_requirements.txt

RUN mkdir tfdragonn
COPY ./*.py ./*.yaml tfdragonn/
