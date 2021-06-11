FROM tiangolo/uwsgi-nginx-flask:python3.7

MAINTAINER maintainer@example.com

RUN apt-get -y update && \
    apt-get install -y vim

ENV UWSGI_INI ./uwsgi.ini

COPY ./app /app
WORKDIR /app

ENV STATIC_URL /static
ENV STATIC_PATH /static

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]

