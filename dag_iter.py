from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

from airflow.utils.dates import days_ago


WORKING_DIRECTORY = "serving"
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
    schedule_interval='*/1 * * * *',
    is_paused_upon_creation=False,
    catchup = False,
    max_active_runs=1
)

dataload = BashOperator(
    # 새로운 데이터 받아오기
    task_id='dataload',
    bash_command=f'python {WORKING_DIRECTORY}/download_data.py', # 임의
    dag=dag,
)

update_data = BashOperator(
    # 새로운 데이터 추가하여 s3에 업로드하기
    task_id='retrain',
    bash_command=f'python {WORKING_DIRECTORY}/upload_s3.py',
    dag=dag,
)

retrain = BashOperator(
    # 재학습
    task_id='retrain',
    bash_command=f'python {WORKING_DIRECTORY}/dkt/train.py --config {CONFIG}',
    dag=dag,
)

packing = BashOperator(
    # 패키징
    task_id='packing',
    bash_command=f'python {WORKING_DIRECTORY}/dkt/packer.py',
    dag=dag
)

drop_reload = BashOperator(
    # 기존의 컨테이너 내리기
    task_id='drop_reload',
    bash_command=f'docker-compose -f {WORKING_DIRECTORY}/docker-compose.yml up -d',
    dag=dag
)

dataload >> update_data >> retrain >> packing >> drop_reload
