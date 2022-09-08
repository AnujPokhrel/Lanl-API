from python:3.10


RUN mkdir /backend

ADD ./requirements.txt /backend/requirements.txt

WORKDIR /backend

RUN pip install -r requirements.txt

EXPOSE 8000
