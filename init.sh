source mentos/bin/activate

kill -9 $(lsof -t -i:6006)

export TZ=Asia/Seoul
export AIRFLOW_HOME=~/airflow
export WORKING_DIRECTORY=$(pwd)

python file_to_db.py --path ${WORKING_DIRECTORY}/event_data.csv

python ${WORKING_DIRECTORY}/db_to_file.py

python ${WORKING_DIRECTORY}/dkt/train.py \
        --model_dir ${WORKING_DIRECTORY}/models \
        --asset_dir ${WORKING_DIRECTORY}/asset \
        --data_dir ${WORKING_DIRECTORY} \
        --file_name data.csv

airflow db init
mkdir ~/airflow/dags

airflow users create \
    --username admin \
    --firstname Junghun \
    --lastname Hwang \
    --password 1234 \
    --role Admin \
    --email anthony9307@naver.com

airflow db reset -y