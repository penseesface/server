FROM nvcr.io/nvidia/tritonserver:21.03-py3

RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip3 install --upgrade --no-cache-dir torch
RUN pip3 install --upgrade --no-cache-dir numpy
RUN pip3 install --upgrade --no-cache-dir opencv-python

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y