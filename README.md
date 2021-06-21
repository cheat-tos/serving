## 📚 Deep Knowledge Tracing(DKT)
**딥러닝을 이용한 지식 상태 추적**(Deep Learning + Knowledge Tracing)으로 특정 시험을 통해 학생의 지식 상태를 파악하고 이를 기반으로 다음 문제를 맞출지 예측하는 태스크입니다.
학습과 망각을 통해 지식 상태는 계속 변화하며 추가되는 문제 풀이 정보로 지식 상태를 지속적으로 추적해야 합니다.

<br>

## 📝 Repository Summary

이 레포지토리는 아래와 같은 아키텍쳐를 구성하기 위한 코드로 이루어져있습니다.
먼저 이 아키텍쳐는 두 개의 서버를 전제한 상태로 구성되었습니다.  

**1. Inference용 서버 : Naver Cloud Platform Server**
- 유저의 Request를 받아 Inference를 수행하고 결과를 렌더링하거나, Model file을 packing하여 새로운 Docker Image로 만드는 작업을 수행합니다.  

**2. Train용 서버 : P40 GPU Server**
- Docker 컨테이너로 구성된 서버로, P40 GPU가 할당되어있습니다. 새로운 데이터를 내려받아 Train을 진행하고, `model.pt` 파일을 생성합니다. Model Train에 GPU 리소스가 많이 필요하기 때문에 해당 서버를 사용했습니다.

### 왜 서버를 나누었는가?
ML/DL Cycle에서 각 Task별로 필요한 리소스의 종류와 수준이 다릅니다. 예를 들어 Train Server의 경우 High-GPU 환경이 세팅되어야 원활한 학습이 가능합니다. 그러나 Inference의 경우 무거운 모델을 올리지 않는다면 굳이 높은 비용을 들여 GPU 서버를 사용할 필요가 없습니다. 이처럼 Task에 따라 비용을 최대한 절약하고 스케일링 가능한 아키텍쳐를 만드는 것이 AI 모델 서빙의 핵심 point 중 하나입니다.

기본적으로 서버 세팅은 NCP Cloud Server와 Docker 컨테이너인 P40 GPU Server에 올라가 있습니다. **따라서 정상적으로 동작하도록 세팅하시려면 Config 또는 코드 내의 경로 설정을 변경해주셔야합니다.**

![](https://i.imgur.com/o9rFiO9.png)


### --- Service Flow

1. 가장 먼저 Inference, Train 각 서버에 git clone을 받고 쉘 스크립트를 이용해 서버 세팅 / 최초의 Client-Inference 컨테이너 생성을 수행합니다.

2. 유저가 Inference 서버에 요청을 보내면, Flask 컨테이너(Client 컨테이너)가 해당 데이터를 Inference 컨테이너(서버 컨테이너)로 전달하여 Score 예측을 수행합니다.  

3. **`update_data_dag`**
    - Inference 과정에서 AWS S3에 업데이트할 데이터를 쌓아두었다가, 주기적으로 S3로 업로드하여 Versioning 합니다. 이 때 **S3 버킷 네임은 반드시 본인의 S3 Bucket Name으로 바꿔주세요.**  


4. **`retrain_and_push_dag`**
    - Train server에서 S3의 업데이트된 데이터를 내려받고, 학습을 수행하여 모델 파일을 생성한 뒤 이를 다시 S3에 업로드하여 Versioning합니다.


5. **`update_inference_dag`**
    - Inference Server에서 모델파일을 내려받고, 이를 기존의 코드와 함께 BentoML로 Packing & Containerizing하여 이미지로 빌드한 후 Docker Hub에 push합니다. 이 때 **현재 Docker Hub 내의 Image Repository 세팅을 본인의 레포지토리로 바꿔주세요.**
    - 각 Inference 컨테이너는 Service 단위로 묶여 Docker Swarm으로 배포됩니다. 마지막으로, Docker Image를 Pull받아 변경점이 생긴 Image를 기준으로 Inference Service를 업데이트합니다.

이후에는 **2-5의 과정을 주기적으로 반복**하여 모델을 재학습하고 업데이트할 수 있습니다.
이 모든 과정은 Apache Airflow Dags에 의해 수행/관리됩니다.

<br>

## 📌 How to run

해당 레포지토리의 최상단에는 두개의 Initialization 쉘 스크립트가 존재합니다.
이 두 쉘 스크립트를 이용하여 각각 Inference Server와 Train Server를 최초 세팅합니다.
이후, Airflow Webserver를 이용하여 Dags를 수행함으로써 서빙 파이프라인을 수행할 수 있습니다.

### `init_for_inference_server.sh`

**🏠 홈 디렉토리** : `/root/`
**✔️ 설치 패키지**
- Docker
- Airflow
- SQLite3

**Description**
- 패키지 세팅 및 Airflow Scheduler 데몬 실행
- GUI Web server 데몬 실행(default port `8080`)
- 최초의 Client-Inference 컨테이너 서비스들을 Docker Swarm으로 Deploy
- 실행 과정에서 aws configure를 설정해야 S3 업로드/다운로드 기능을 정상적으로 수행 가능

### `init_for_train_server.sh`

**🏠 홈 디렉토리** : `/opt/ml/`
**✔️ 설치 패키지**
- Docker
- Airflow
- SQLite3

**Description**
- 패키지 세팅 및 Airflow Scheduler 데몬 실행
- GUI Web server 데몬 실행(default port `6006`)
- 실행 과정에서 aws configure를 설정해야 S3 업로드/다운로드 기능을 정상적으로 수행 가능

### Airflow Dags

**아래 Dag들을 차례대로 수행하셨을 때 한 Cycle이 완성됩니다.**

#### `update_data_dag` - Inference Server

Inference 과정에서 쌓인 유저 Interaction 로그들을 기존 데이터와 합쳐서 S3에 업로드합니다.

#### `retrain_and_push_dag` - Train Server

S3에 업데이트된 유저 Interaction 로그들을 불러와 재학습된 모델을 S3에 업로드 합니다.

#### `update_inference_server_dag` - Inference Server

S3에 저장된 Retrain 된 모델을 Inference server로 읽어와 rolling update를 진행합니다.

<br>

## 🛠️ Installation

### Server Dependencies (pip3)
- easydict==1.9
- numpy==1.19.5
- pandas==1.1.5
- sklearn==0.0
- torch==1.6.0
- transformers==4.6.1
- bentoml==0.12.1
- boto3==1.17.78
- apache-airflow
- sqlalchemy < 1.4.0
- attrdict

<br>

## 🏛️ File Structure

```
├── client                                 # Components of Flask
│   ├── Dockerfile
│   ├── app
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   ├── static
│   │   │   └── ...
│   │   ├── templates
│   │   │   └── index.html
│   │   └── uwsgi.ini
│   └── start.sh                           # Build and Run client server container
│
├── dags                                   # Airflow DAGs
│   ├── retrain_and_push_dag.py
│   ├── update_data_dag.py
│   └── update_inference_server_dag.py
│
├── data                                   # Data WH upload/download & save data
│   └── ...
├── models                                 # Data WH upload/download & save model
│   └── ...
│
├── docker-compose.yml
├── docker_manual
│   ├── docker_commands.sh                 # Summary of docker commands
│   └── service-init.sh                    # Start inference with docker swarm
│
├── config                                 # Model Config JSON Files 
│   └── ...
├── asset                                  # Encoder class npy files 
│   └── ..
├── dkt                                    # Baseline codes
│   └── ...                                
├── args.py                                # Get user arguments
├── train.py                               # Training model
├── requirements.txt
│
├── questions.csv
├── inference.py                           # Inferenece using question.csv
├── packer.py                              # Packing model, encoders to bentoml service class
├── service.py                             # Compose inference api 
│
├── init_for_inference_server.sh           # Initialize for Inference server
├── init_for_train_server.sh               # Initialize for Train server
└── README.md
```

### `client`

Flask를 이용하여 웹페이지를 구성하는 요소들입니다.
Interaction하는 화면을 바꾸고싶다면 이 파트를 수정하시면 됩니다.

### `dags`

Airflow에 Dag로 등록될 파일들이 존재하는 디렉토리입니다.
`cp` 커맨드를 이용하여 initialization 과정에서 `airflow/dags/` 디렉토리로 복사됩니다.

### `data`

데이터가 유저 Interaction을 통해 새롭게 추가되었을 때, S3에 Upload/Download를 처리합니다.
또, Upload할 csv 데이터 / Download한 csv 데이터를 save합니다.

### `models`

모델이 새롭게 추가되었을 때, S3에 Upload/Download를 처리합니다.
또, Upload할 모델 / Download한 모델을 save합니다.

### `docker_manual`

Docker 명령어를 정리해둔 `docker_commands.sh`과 초기 Inference 컨테이너 서비스 세팅을 위한 `service_init.sh`로 이루어져있습니다.

### `config`

Train 및 Inference를 위한 argument configuration 파일들로 이루어져있습니다.
	

### `asset`

Training 과정에서 Categorical data를 변형하는 Encoder 정보가 npy 파일들로 이루어져 있습니다.

### `dkt`

DKT Task 수행을 위한 Trainer, Model, Metric, Loss 등의 코드가 있는 baseline code 디렉토리입니다.

<br>

## 👪 Contributor

> 김성익 [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/SeongIkKim) [![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white)](mailto:kpic1638@gmail.com)

> 김동우 [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github&link=https://zzsza.github.io/)](https://github.com/ddooom)

> 황정훈 [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/wjdgns7712) [![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white)](mailto:wjdgns7712@gmail.com)
