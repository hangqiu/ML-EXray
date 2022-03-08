FROM tensorflow/tensorflow:latest-gpu
RUN useradd -rm -d /home/mlexray -s /bin/bash -g root -G sudo -p mlexray mlexray
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

#USER mlexray
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -i https://test.pypi.org/simple/ MLEXray

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

#WORKDIR /home/mlexray

COPY data data
COPY model model
RUN mkdir trace
COPY ClassificationRunnerTest.py .

EXPOSE 5000

#USER root
#CMD ["python3", "ClassificationRunnerTest.py"]