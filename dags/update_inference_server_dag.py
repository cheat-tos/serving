from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

from airflow.utils.dates import days_ago

# THIS DAG IS FOR INFERENCE SERVER.
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
    'update_inference_server_dag',
    default_args=default_args,
    description='Replace inference server with recent image',
    start_date=days_ago(0),
    schedule_interval=None, # externally triggered
    # schedule_interval='0 3 * * * *', # daily update at 3 (maybe after model train is completed)'
    is_paused_upon_creation=True,
    catchup = False,
    max_active_runs=1
)

load_model = BashOperator(
    # 모델 받아오기
    task_id='load_model',
    bash_command=f'python3 {WORKING_DIRECTORY}/models/load_model_in_inference.py',
    dag=dag
)

packing = BashOperator(
    # Bento Service 패키징
    task_id='packing',
    bash_command=f'python3 {WORKING_DIRECTORY}/packer.py',
    dag=dag
)

build_and_push = BashOperator(
    # docker containzerize
    task_id='build_and_push_image',
    bash_command=f"""
    bentoml containerize PytorchDKT:latest -t kpic5014/bento-dkt:latest
    docker push kpic5014/bento-dkt
    """,
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
      dkt-service_inference',
    dag=dag
)

load_model >> packing >> build_and_push >> rolling_update
