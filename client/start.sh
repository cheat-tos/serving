#!/bin/bash
app="dkt-client-flask"
docker build -t ${app} .
docker run -d -p 6006:6006 \
  --name=${app} \
  -v $PWD/app ${app}
