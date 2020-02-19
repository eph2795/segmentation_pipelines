FROM pytorch/pytorch:latest

WORKDIR /inference

COPY stack_segmentation /inference/stack_segmentation
COPY inference.py /inference/inference.py
COPY docker_requirements.txt /inference/docker_requirements.txt

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y libffi-dev python-dev build-essential
RUN apt-get install -y libgtk2.0-dev
RUN pip install -r docker_requirements.txt