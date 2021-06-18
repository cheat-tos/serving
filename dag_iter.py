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
    # schedule_interval='0 2 * * * *', # daily update at 2
    is_paused_upon_creation=False,
    catchup = False,
    max_active_runs=1
)

load_data = BashOperator(
    # 새로운 데이터 받아오기
    task_id='load_data',
    bash_command=f'python3 /root/{WORKING_DIRECTORY}/data/load_data.py', # 임의
    dag=dag,
)

update_data = BashOperator(
    # 새로운 데이터 추가하여 s3에 업로드하기
    task_id='update_data',
    bash_command=f'python3 /root/{WORKING_DIRECTORY}/data/update_data.py',
    dag=dag,
)

retrain = BashOperator(
    # 재학습
    task_id='retrain',
    bash_command=f'python3 /root/{WORKING_DIRECTORY}/train.py --config {CONFIG}',
    dag=dag,
)

packing = BashOperator(
    # 패키징
    task_id='packing',
    bash_command=f'python3 /root/{WORKING_DIRECTORY}/packer.py',
    dag=dag
)

rolling_update = BashOperator(
    # 기존의 컨테이너 내리기
    task_id='rolling_update',
    bash_command=f'docker service update \
      --update-parallelism 1 \
      --update-delay 10s \
      --image kpic5014/bento-dkt:latest \
      --detach=false \
      dkt-service_client',
    dag=dag
)

load_data >> update_data >> retrain >> packing >> rolling_update
