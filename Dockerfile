FROM pytorch/pytorch:latest

WORKDIR /inference

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y libffi-dev python-dev build-essential
RUN apt-get install -y libgtk2.0-dev

COPY docker_requirements.txt /inference/docker_requirements.txt
RUN pip install -r docker_requirements.txt

COPY docker_code/count.py /inference/count.py
COPY docker_code/inference.py /inference/inference.py
COPY docker_code/args_parse.py /inference/args_parse.py
COPY docker_code/stack_segmentation /inference/stack_segmentation

