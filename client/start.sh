#!/bin/bash
app="dkt-client-flask"
docker build -t ${app} .
docker run -p 6006:6006 \
  --name=${app} --network bento_serving_mentos ${app}
# for volumne mount(access to host storage)
#  -v ${PWD}/app:/app ${app}
