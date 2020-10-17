FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter
LABEL EEG JUPYTER


ADD requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


