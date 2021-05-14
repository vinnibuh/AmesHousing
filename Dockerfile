FROM python:3.8.10-slim-buster
MAINTAINER Ivchenkov Yaroslav "ivchenkov.yap@phystech.edu"

WORKDIR /app

RUN apt-get update
RUN apt-get -y install make

RUN pip3 install pipenv
COPY Pipfile ./
COPY Pipfile.lock ./
RUN set -ex && pipenv install --system --deploy 

COPY . .
