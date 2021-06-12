#!/bin/bash
app="dkt-client-flask"
docker build -t ${app} .
docker run -d -p 6006:6006 \
  --name=${app} ${app}
# for volumne mount(access to host storage)
#  -v ${PWD}/app:/app ${app}
