# Use a base image with Python 3.9
FROM python:3.9

# Install system dependencies (if necessary)
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev libcairo2-dev pkg-config python3.9-dev libcairo2 ffmpeg libsm6 libxext6

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py

# Install Python packages
RUN pip3.9 install --upgrade pip wheel setuptools --no-cache-dir && \ 
    pip3.9 install --no-cache-dir torch torchvision numpy matplotlib scikit-learn Pillow

WORKDIR /app

