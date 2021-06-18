from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

from airflow.utils.dates import days_ago


WORKING_DIRECTORY = "/root/serving"
CONFIG = "lstm"


default_args = {
    'owner': 'mentos',
    'email': ['mentos@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}


dag = DAG(
    'dag_iter',
    default_args=default_args,
    description='DKT workflow management',
    start_date=days_ago(0),
    # schedule_interval='0 2 * * * *', # daily update at 2
    is_paused_upon_creation=False,
    catchup = False,
    max_active_runs=1
)

load_data = BashOperator(
    # 새로운 데이터 받아오기
    task_id='load_data',
    bash_command=f'python3 {WORKING_DIRECTORY}/data/load_data.py', # 임의
    dag=dag,
)

update_data = BashOperator(
    # 새로운 데이터 추가하여 s3에 업로드하기
    task_id='update_data',
    bash_command=f'python3 {WORKING_DIRECTORY}/data/update_data.py',
    dag=dag,
)

load_data >> update_data >> rolling_update
