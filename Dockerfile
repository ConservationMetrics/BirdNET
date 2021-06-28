# Build from Python 3.7 slim
FROM python:3.7-slim

# Install required packages while keeping the image small
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg build-essential gfortran libblas-dev liblapack-dev && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install numpy scipy librosa future
RUN pip install python-dateutil

# Install Theano and Lasagne
RUN pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
RUN pip install https://github.com/Lasagne/Lasagne/archive/master.zip

# Import all scripts
COPY . ./

WORKDIR .
