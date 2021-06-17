## How to run

```bash
  # install packages
  $ sudo yum install git
  $ sudo yum install docker
  $ sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" \
            -o /usr/local/bin/docker-compose
  
  # assign permission
  $ sudo chmod +x /usr/local/bin/docker-compose
  $ sudo usermod -aG docker $USER
  $ newgrp docker
  $ sudo systemctl restart docker
  
  # pull repo
  $ git clone https://github.com/cheat-tos/serving.git
  
  # execute
  $ cd serving
  $ docker-compose up
```

## envirment
```
system-release-2-13.amzn2.x86_64
```

## Deep Knowledge Tracing(DKT)
<b>딥러닝을 이용한 지식 상태 추적(Deep Learning + Knowledge Tracing)</b>으로 특정 시험을 통해 학생의 지식 상태를 파악하고 이를 기반으로 다음 문제를 맞출지 예측하는 테스크입니다.

학습과 망각을 통해 지식 상태는 계속 변화하며 추가되는 문제 풀이 정보로 지식 상태를 지속적으로 추적해야 합니다.

## Installation
### Dependencies
- easydict==1.9
- numpy==1.19.5
- pandas==1.2.4
- sklearn==0.0
- torch==1.6.0
- transformers==4.6.1
- bentoml==0.12.1
- boto3==1.17.78
```
pip install -r requirements.txt
```

## File Structure
### bento_serve
```
bento_serve
|
├── client
|   ├── Dockerfile
|   ├── app
|   |   ├── main.py
|   |   ├── requirements.txt
|   |   ├── static
|   |   |   └── Number-Rolling-Animation-jQuery-numberAnimate
|   |   |       ├── README.md
|   |   |       ├── numberAnimate.js
|   |   |       ├── simpleExample.html
|   |   |       └── timeExample.html
|   |   ├── templates
|   |   |   └── index.html
|   |   └── uwsgi.ini
|   └── start.sh
|
├── data
|   ├── load.py
|   └── update.py
|
├── config
|   ├── lstm.json
|   └── saint.json
|
├── dkt
|   ├── __init__.py
|   ├── criterion.py
|   ├── dataloader.py
|   ├── metric.py
|   ├── model.py
|   ├── optimizer.py
|   ├── scheduler.py
|   ├── trainer.py
|   └── utils.py
|
├── args.py
├── inference.py
├── packer.py
├── questions.csv
├── requirements.txt
├── service.py
└── train.py
 

├── docker-compose.yml
├── docker_manual
|   └── docker_build_and_push.sh
├── dag_iter.py
├── dag_static.py
```
