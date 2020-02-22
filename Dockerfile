FROM pytorch/pytorch:latest

WORKDIR /segmentation_pipelines

RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y libffi-dev python-dev build-essential
RUN apt-get install -y libgtk2.0-dev

COPY docker_requirements.txt /segmentation_pipelines/docker_requirements.txt
RUN pip install -r docker_requirements.txt

COPY docker_code /segmentation_pipelines/docker_code
