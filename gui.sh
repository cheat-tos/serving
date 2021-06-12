source mentos/bin/activate

export TZ=Asia/Seoul
export AIRFLOW_HOME=~/airflow
export WORKING_DIRECTORY=$(pwd)

airflow webserver -p 6006