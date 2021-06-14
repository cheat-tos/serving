FROM tiangolo/uwsgi-nginx-flask:python3.7

MAINTAINER maintainer@example.com

ENV NGINX_WORKER_PROCESS auto

RUN apt-get -y update && \
    apt-get install -y vim

ENV UWSGI_INI /app/uwsgi.ini

COPY ./app /app
WORKDIR /app

ENV STATIC_URL /static
ENV STATIC_PATH /app/static

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# It does not explicitly limit access, but implicitly informs
# https://stackoverflow.com/questions/22111060/what-is-the-difference-between-expose-and-publish-in-docker
# ENV LISTEN_PORT 6006
# EXPOSE 6006

CMD ["uwsgi", "uwsgi.ini"]
ENTRYPOINT ["python", "main.py"]

