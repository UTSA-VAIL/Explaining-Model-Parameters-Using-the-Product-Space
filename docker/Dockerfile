# BASE CONTAINER
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# TIME ZONE FOR TZDATA
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# APT INSTALLS
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev git 
RUN apt -y install libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt -y install python3-pip

# PIP
RUN python3 -m pip install --upgrade pip
COPY ./requirements.txt /tmp
RUN python3 -m pip install -r /tmp/requirements.txt

# APP DIRECTORY PATH
WORKDIR /app
