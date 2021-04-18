FROM python:3.9.3-slim-buster
MAINTAINER Ivchenkov Yaroslav "ivchenkov.yap@phystech.edu"
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
RUN python3 setup.py install
ENTRYPOINT ["/bin/sh"]
