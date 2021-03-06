## ๐ Deep Knowledge Tracing(DKT)
**๋ฅ๋ฌ๋์ ์ด์ฉํ ์ง์ ์ํ ์ถ์ **(Deep Learning + Knowledge Tracing)์ผ๋ก ํน์  ์ํ์ ํตํด ํ์์ ์ง์ ์ํ๋ฅผ ํ์ํ๊ณ  ์ด๋ฅผ ๊ธฐ๋ฐ์ผ๋ก ๋ค์ ๋ฌธ์ ๋ฅผ ๋ง์ถ์ง ์์ธกํ๋ ํ์คํฌ์๋๋ค.
ํ์ต๊ณผ ๋ง๊ฐ์ ํตํด ์ง์ ์ํ๋ ๊ณ์ ๋ณํํ๋ฉฐ ์ถ๊ฐ๋๋ ๋ฌธ์  ํ์ด ์ ๋ณด๋ก ์ง์ ์ํ๋ฅผ ์ง์์ ์ผ๋ก ์ถ์ ํด์ผ ํฉ๋๋ค.

<br>

## ๐ Repository Summary

์ด ๋ ํฌ์งํ ๋ฆฌ๋ ์๋์ ๊ฐ์ ์ํคํ์ณ๋ฅผ ๊ตฌ์ฑํ๊ธฐ ์ํ ์ฝ๋๋ก ์ด๋ฃจ์ด์ ธ์์ต๋๋ค.
๋จผ์  ์ด ์ํคํ์ณ๋ ๋ ๊ฐ์ ์๋ฒ๋ฅผ ์ ์ ํ ์ํ๋ก ๊ตฌ์ฑ๋์์ต๋๋ค.  

**1. Inference์ฉ ์๋ฒ : Naver Cloud Platform Server**
- ์ ์ ์ Request๋ฅผ ๋ฐ์ Inference๋ฅผ ์ํํ๊ณ  ๊ฒฐ๊ณผ๋ฅผ ๋ ๋๋งํ๊ฑฐ๋, Model file์ packingํ์ฌ ์๋ก์ด Docker Image๋ก ๋ง๋๋ ์์์ ์ํํฉ๋๋ค.  

**2. Train์ฉ ์๋ฒ : P40 GPU Server**
- Docker ์ปจํ์ด๋๋ก ๊ตฌ์ฑ๋ ์๋ฒ๋ก, P40 GPU๊ฐ ํ ๋น๋์ด์์ต๋๋ค. ์๋ก์ด ๋ฐ์ดํฐ๋ฅผ ๋ด๋ ค๋ฐ์ Train์ ์งํํ๊ณ , `model.pt` ํ์ผ์ ์์ฑํฉ๋๋ค. Model Train์ GPU ๋ฆฌ์์ค๊ฐ ๋ง์ด ํ์ํ๊ธฐ ๋๋ฌธ์ ํด๋น ์๋ฒ๋ฅผ ์ฌ์ฉํ์ต๋๋ค.

### ์ ์๋ฒ๋ฅผ ๋๋์๋๊ฐ?
ML/DL Cycle์์ ๊ฐ Task๋ณ๋ก ํ์ํ ๋ฆฌ์์ค์ ์ข๋ฅ์ ์์ค์ด ๋ค๋ฆ๋๋ค. ์๋ฅผ ๋ค์ด Train Server์ ๊ฒฝ์ฐ High-GPU ํ๊ฒฝ์ด ์ธํ๋์ด์ผ ์ํํ ํ์ต์ด ๊ฐ๋ฅํฉ๋๋ค. ๊ทธ๋ฌ๋ Inference์ ๊ฒฝ์ฐ ๋ฌด๊ฑฐ์ด ๋ชจ๋ธ์ ์ฌ๋ฆฌ์ง ์๋๋ค๋ฉด ๊ตณ์ด ๋์ ๋น์ฉ์ ๋ค์ฌ GPU ์๋ฒ๋ฅผ ์ฌ์ฉํ  ํ์๊ฐ ์์ต๋๋ค. ์ด์ฒ๋ผ Task์ ๋ฐ๋ผ ๋น์ฉ์ ์ต๋ํ ์ ์ฝํ๊ณ  ์ค์ผ์ผ๋ง ๊ฐ๋ฅํ ์ํคํ์ณ๋ฅผ ๋ง๋๋ ๊ฒ์ด AI ๋ชจ๋ธ ์๋น์ ํต์ฌ point ์ค ํ๋์๋๋ค.

๊ธฐ๋ณธ์ ์ผ๋ก ์๋ฒ ์ธํ์ NCP Cloud Server์ Docker ์ปจํ์ด๋์ธ P40 GPU Server์ ์ฌ๋ผ๊ฐ ์์ต๋๋ค. **๋ฐ๋ผ์ ์ ์์ ์ผ๋ก ๋์ํ๋๋ก ์ธํํ์๋ ค๋ฉด Config ๋๋ ์ฝ๋ ๋ด์ ๊ฒฝ๋ก ์ค์ ์ ๋ณ๊ฒฝํด์ฃผ์์ผํฉ๋๋ค.**

![](https://i.imgur.com/o9rFiO9.png)


### --- Service Flow

1. ๊ฐ์ฅ ๋จผ์  Inference, Train ๊ฐ ์๋ฒ์ git clone์ ๋ฐ๊ณ  ์ ์คํฌ๋ฆฝํธ๋ฅผ ์ด์ฉํด ์๋ฒ ์ธํ / ์ต์ด์ Client-Inference ์ปจํ์ด๋ ์์ฑ์ ์ํํฉ๋๋ค.

2. ์ ์ ๊ฐ Inference ์๋ฒ์ ์์ฒญ์ ๋ณด๋ด๋ฉด, Flask ์ปจํ์ด๋(Client ์ปจํ์ด๋)๊ฐ ํด๋น ๋ฐ์ดํฐ๋ฅผ Inference ์ปจํ์ด๋(์๋ฒ ์ปจํ์ด๋)๋ก ์ ๋ฌํ์ฌ Score ์์ธก์ ์ํํฉ๋๋ค.  

3. **`update_data_dag`**
    - Inference ๊ณผ์ ์์ AWS S3์ ์๋ฐ์ดํธํ  ๋ฐ์ดํฐ๋ฅผ ์์๋์๋ค๊ฐ, ์ฃผ๊ธฐ์ ์ผ๋ก S3๋ก ์๋ก๋ํ์ฌ Versioning ํฉ๋๋ค. ์ด ๋ **S3 ๋ฒํท ๋ค์์ ๋ฐ๋์ ๋ณธ์ธ์ S3 Bucket Name์ผ๋ก ๋ฐ๊ฟ์ฃผ์ธ์.**  


4. **`retrain_and_push_dag`**
    - Train server์์ S3์ ์๋ฐ์ดํธ๋ ๋ฐ์ดํฐ๋ฅผ ๋ด๋ ค๋ฐ๊ณ , ํ์ต์ ์ํํ์ฌ ๋ชจ๋ธ ํ์ผ์ ์์ฑํ ๋ค ์ด๋ฅผ ๋ค์ S3์ ์๋ก๋ํ์ฌ Versioningํฉ๋๋ค.


5. **`update_inference_dag`**
    - Inference Server์์ ๋ชจ๋ธํ์ผ์ ๋ด๋ ค๋ฐ๊ณ , ์ด๋ฅผ ๊ธฐ์กด์ ์ฝ๋์ ํจ๊ป BentoML๋ก Packing & Containerizingํ์ฌ ์ด๋ฏธ์ง๋ก ๋น๋ํ ํ Docker Hub์ pushํฉ๋๋ค. ์ด ๋ **ํ์ฌ Docker Hub ๋ด์ Image Repository ์ธํ์ ๋ณธ์ธ์ ๋ ํฌ์งํ ๋ฆฌ๋ก ๋ฐ๊ฟ์ฃผ์ธ์.**
    - ๊ฐ Inference ์ปจํ์ด๋๋ Service ๋จ์๋ก ๋ฌถ์ฌ Docker Swarm์ผ๋ก ๋ฐฐํฌ๋ฉ๋๋ค. ๋ง์ง๋ง์ผ๋ก, Docker Image๋ฅผ Pull๋ฐ์ ๋ณ๊ฒฝ์ ์ด ์๊ธด Image๋ฅผ ๊ธฐ์ค์ผ๋ก Inference Service๋ฅผ ์๋ฐ์ดํธํฉ๋๋ค.

์ดํ์๋ **2-5์ ๊ณผ์ ์ ์ฃผ๊ธฐ์ ์ผ๋ก ๋ฐ๋ณต**ํ์ฌ ๋ชจ๋ธ์ ์ฌํ์ตํ๊ณ  ์๋ฐ์ดํธํ  ์ ์์ต๋๋ค.
์ด ๋ชจ๋  ๊ณผ์ ์ Apache Airflow Dags์ ์ํด ์ํ/๊ด๋ฆฌ๋ฉ๋๋ค.

<br>

## ๐ How to run

ํด๋น ๋ ํฌ์งํ ๋ฆฌ์ ์ต์๋จ์๋ ๋๊ฐ์ Initialization ์ ์คํฌ๋ฆฝํธ๊ฐ ์กด์ฌํฉ๋๋ค.
์ด ๋ ์ ์คํฌ๋ฆฝํธ๋ฅผ ์ด์ฉํ์ฌ ๊ฐ๊ฐ Inference Server์ Train Server๋ฅผ ์ต์ด ์ธํํฉ๋๋ค.
์ดํ, Airflow Webserver๋ฅผ ์ด์ฉํ์ฌ Dags๋ฅผ ์ํํจ์ผ๋ก์จ ์๋น ํ์ดํ๋ผ์ธ์ ์ํํ  ์ ์์ต๋๋ค.

### `init_for_inference_server.sh`

**๐  ํ ๋๋ ํ ๋ฆฌ** : `/root/`
**โ๏ธ ์ค์น ํจํค์ง**
- Docker
- Airflow
- SQLite3

**Description**
- ํจํค์ง ์ธํ ๋ฐ Airflow Scheduler ๋ฐ๋ชฌ ์คํ
- GUI Web server ๋ฐ๋ชฌ ์คํ(default port `8080`)
- ์ต์ด์ Client-Inference ์ปจํ์ด๋ ์๋น์ค๋ค์ Docker Swarm์ผ๋ก Deploy
- ์คํ ๊ณผ์ ์์ aws configure๋ฅผ ์ค์ ํด์ผ S3 ์๋ก๋/๋ค์ด๋ก๋ ๊ธฐ๋ฅ์ ์ ์์ ์ผ๋ก ์ํ ๊ฐ๋ฅ

### `init_for_train_server.sh`

**๐  ํ ๋๋ ํ ๋ฆฌ** : `/opt/ml/`
**โ๏ธ ์ค์น ํจํค์ง**
- Docker
- Airflow
- SQLite3

**Description**
- ํจํค์ง ์ธํ ๋ฐ Airflow Scheduler ๋ฐ๋ชฌ ์คํ
- GUI Web server ๋ฐ๋ชฌ ์คํ(default port `6006`)
- ์คํ ๊ณผ์ ์์ aws configure๋ฅผ ์ค์ ํด์ผ S3 ์๋ก๋/๋ค์ด๋ก๋ ๊ธฐ๋ฅ์ ์ ์์ ์ผ๋ก ์ํ ๊ฐ๋ฅ

### Airflow Dags

**์๋ Dag๋ค์ ์ฐจ๋ก๋๋ก ์ํํ์จ์ ๋ ํ Cycle์ด ์์ฑ๋ฉ๋๋ค.**

#### `update_data_dag` - Inference Server

Inference ๊ณผ์ ์์ ์์ธ ์ ์  Interaction ๋ก๊ทธ๋ค์ ๊ธฐ์กด ๋ฐ์ดํฐ์ ํฉ์ณ์ S3์ ์๋ก๋ํฉ๋๋ค.

#### `retrain_and_push_dag` - Train Server

S3์ ์๋ฐ์ดํธ๋ ์ ์  Interaction ๋ก๊ทธ๋ค์ ๋ถ๋ฌ์ ์ฌํ์ต๋ ๋ชจ๋ธ์ S3์ ์๋ก๋ ํฉ๋๋ค.

#### `update_inference_server_dag` - Inference Server

S3์ ์ ์ฅ๋ Retrain ๋ ๋ชจ๋ธ์ Inference server๋ก ์ฝ์ด์ rolling update๋ฅผ ์งํํฉ๋๋ค.

<br>

## ๐ ๏ธ Installation

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

## ๐๏ธ File Structure

```
โโโ client                                 # Components of Flask
โย ย  โโโ Dockerfile
โย ย  โโโ app
โย ย  โย ย  โโโ main.py
โย ย  โย ย  โโโ requirements.txt
โย ย  โย ย  โโโ static
โย ย  โย ย  โย ย  โโโ ...
โย ย  โย ย  โโโ templates
โย ย  โย ย  โย ย  โโโ index.html
โย ย  โย ย  โโโ uwsgi.ini
โย ย  โโโ start.sh                           # Build and Run client server container
โ
โโโ dags                                   # Airflow DAGs
โย ย  โโโ retrain_and_push_dag.py
โย ย  โโโ update_data_dag.py
โย ย  โโโ update_inference_server_dag.py
โ
โโโ data                                   # Data WH upload/download & save data
โย ย  โโโ ...
โโโ models                                 # Data WH upload/download & save model
โย ย  โโโ ...
โ
โโโ docker-compose.yml
โโโ docker_manual
โย ย  โโโ docker_commands.sh                 # Summary of docker commands
โย ย  โโโ service-init.sh                    # Start inference with docker swarm
โ
โโโ config                                 # Model Config JSON Files 
โย ย  โโโ ...
โโโ asset                                  # Encoder class npy files 
โย ย  โโโ ..
โโโ dkt                                    # Baseline codes
โย ย  โโโ ...                                
โโโ args.py                                # Get user arguments
โโโ train.py                               # Training model
โโโ requirements.txt
โ
โโโ questions.csv
โโโ inference.py                           # Inferenece using question.csv
โโโ packer.py                              # Packing model, encoders to bentoml service class
โโโ service.py                             # Compose inference api 
โ
โโโ init_for_inference_server.sh           # Initialize for Inference server
โโโ init_for_train_server.sh               # Initialize for Train server
โโโ README.md
```

### `client`

Flask๋ฅผ ์ด์ฉํ์ฌ ์นํ์ด์ง๋ฅผ ๊ตฌ์ฑํ๋ ์์๋ค์๋๋ค.
Interactionํ๋ ํ๋ฉด์ ๋ฐ๊พธ๊ณ ์ถ๋ค๋ฉด ์ด ํํธ๋ฅผ ์์ ํ์๋ฉด ๋ฉ๋๋ค.

### `dags`

Airflow์ Dag๋ก ๋ฑ๋ก๋  ํ์ผ๋ค์ด ์กด์ฌํ๋ ๋๋ ํ ๋ฆฌ์๋๋ค.
`cp` ์ปค๋งจ๋๋ฅผ ์ด์ฉํ์ฌ initialization ๊ณผ์ ์์ `airflow/dags/` ๋๋ ํ ๋ฆฌ๋ก ๋ณต์ฌ๋ฉ๋๋ค.

### `data`

๋ฐ์ดํฐ๊ฐ ์ ์  Interaction์ ํตํด ์๋กญ๊ฒ ์ถ๊ฐ๋์์ ๋, S3์ Upload/Download๋ฅผ ์ฒ๋ฆฌํฉ๋๋ค.
๋, Uploadํ  csv ๋ฐ์ดํฐ / Downloadํ csv ๋ฐ์ดํฐ๋ฅผ saveํฉ๋๋ค.

### `models`

๋ชจ๋ธ์ด ์๋กญ๊ฒ ์ถ๊ฐ๋์์ ๋, S3์ Upload/Download๋ฅผ ์ฒ๋ฆฌํฉ๋๋ค.
๋, Uploadํ  ๋ชจ๋ธ / Downloadํ ๋ชจ๋ธ์ saveํฉ๋๋ค.

### `docker_manual`

Docker ๋ช๋ น์ด๋ฅผ ์ ๋ฆฌํด๋ `docker_commands.sh`๊ณผ ์ด๊ธฐ Inference ์ปจํ์ด๋ ์๋น์ค ์ธํ์ ์ํ `service_init.sh`๋ก ์ด๋ฃจ์ด์ ธ์์ต๋๋ค.

### `config`

Train ๋ฐ Inference๋ฅผ ์ํ argument configuration ํ์ผ๋ค๋ก ์ด๋ฃจ์ด์ ธ์์ต๋๋ค.
	

### `asset`

Training ๊ณผ์ ์์ Categorical data๋ฅผ ๋ณํํ๋ Encoder ์ ๋ณด๊ฐ npy ํ์ผ๋ค๋ก ์ด๋ฃจ์ด์ ธ ์์ต๋๋ค.

### `dkt`

DKT Task ์ํ์ ์ํ Trainer, Model, Metric, Loss ๋ฑ์ ์ฝ๋๊ฐ ์๋ baseline code ๋๋ ํ ๋ฆฌ์๋๋ค.

<br>

## ๐ช Contributor

> ๊น์ฑ์ต [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/SeongIkKim) [![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white)](mailto:kpic1638@gmail.com)

> ๊น๋์ฐ [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github&link=https://zzsza.github.io/)](https://github.com/ddooom)

> ํฉ์ ํ [![Git Badge](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github)](https://github.com/wjdgns7712) [![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white)](mailto:wjdgns7712@gmail.com)
