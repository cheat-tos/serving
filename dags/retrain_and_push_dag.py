from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

from airflow.utils.dates import days_ago


WORKING_DIRECTORY = "/opt/ml/serving"
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
    'retrain_and_push_dag',
    default_args=default_args,
    description='Retrain, Packing, Push ml service in GPU server',
    start_date=days_ago(0),
    # schedule_interval='0 2 * * * *', # daily update at 2
    is_paused_upon_creation=False,
    catchup = False,
    max_active_runs=1
)

load_data = BashOperator(
    # 새로운 데이터 받아오기
    task_id='load_data',
    bash_command=f'python3 {WORKING_DIRECTORY}/data/load_data_in_train.py', # 임의
    dag=dag,
)

retrain = BashOperator(
    # 재학습
    task_id='retrain',
    bash_command=f'python3 {WORKING_DIRECTORY}/train.py --config {CONFIG}',
    dag=dag,
)

upload_model = BashOperator(
    # 모델 s3에 업로드
    task_id='upload model to s3',
    bash_command=f'python3 {WORKING_DIRECTORY}/models/upload_model_in_train.py',
    dag=dag
)

load_data >> retrain >> upload_model
