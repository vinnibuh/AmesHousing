FROM python:3.9.3-slim-buster
MAINTAINER Ivchenkov Yaroslav "ivchenkov.yap@phystech.edu"
WORKDIR /app
COPY requirements-test.txt requirements-test.txt
RUN apt-get update
RUN pip3 install -r requirements-test.txt
COPY . .
RUN python3 setup.py install
