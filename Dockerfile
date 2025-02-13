FROM ubuntu:jammy-20230126

WORKDIR /workspace

ENV TZ="Europe/Ljubljana"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libc6-dev-i386 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    vim \
    zsh \
    git \
    wget \
    sudo \
    unzip \
    zip \
    locales \
    lsb-core \
    python3 \
    python-is-python3 \
    python3-pip \
    python3-setuptools-scm \
    python3-venv \
    python3-tk \
    verilator
RUN locale-gen "en_US.UTF-8"

COPY requirements.txt /tmp/requirements.txt
SHELL ["/bin/bash", "-c"] 
RUN python -m pip install --upgrade pip
RUN python -m venv /venv/ && \
    source /venv/bin/activate && \
    pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
ENV PATH="/venv/bin:/workspace:$PATH"
env PYTHONPATH="/workspace"
RUN wget -P /c4ml/ https://github.com/cs-jsi/chisel4ml/releases/download/0.3.6/chisel4ml.jar
RUN apt install default-jre cmake -y
ENV LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
