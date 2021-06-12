source mentos/bin/activate

export TZ=Asia/Seoul
export AIRFLOW_HOME=~/airflow
export WORKING_DIRECTORY=$(pwd)

python ${WORKING_DIRECTORY}/server/server.py \
        --port 6007