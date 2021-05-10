FROM python:3.9.3-slim-buster
MAINTAINER Ivchekov Yaroslav "ivchenkov.yap@phystech.edu"
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
