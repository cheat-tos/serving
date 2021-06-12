from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

from airflow.utils.dates import days_ago



default_args = {
    'owner': 'Peter Parker',
    'email': ['example@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}





dag = DAG(
    'dkt_pipeline',
    default_args=default_args,
    description='DKT workflow management',
    start_date=days_ago(0),
    schedule_interval='*/1 * * * *',
    is_paused_upon_creation=False,
    catchup = False,
    max_active_runs=1
)

dataload = BashOperator(
    task_id='dataload',
    bash_command='python ${WORKING_DIRECTORY}/db_to_file.py',
    dag=dag,
)

train = BashOperator(
    task_id='train',
    bash_command='python ${WORKING_DIRECTORY}/dkt/train.py \
                    --model_dir ${WORKING_DIRECTORY}/models \
                    --asset_dir ${WORKING_DIRECTORY}/asset \
                    --data_dir ${WORKING_DIRECTORY} --file_name data.csv',
    dag=dag,
)

reload = BashOperator(
    task_id='reload',
    bash_command="touch -m ${WORKING_DIRECTORY}/server/server.py",
    dag=dag,
)


dataload >> train >> reload
