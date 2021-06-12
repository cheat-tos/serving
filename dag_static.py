from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

from airflow.utils.dates import days_ago



default_args = {
    'owner': 'abc',
    'email': ['abcabc@abc.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}


dag = DAG(
    'dag_static',
    default_args=default_args,
    description='DKT workflow management',
    start_date=days_ago(0),
    is_paused_upon_creation=False,
    catchup = False,
    max_active_runs=1
)

load = BashOperator(
    # 초기학습 된 컨테이너 올리기
    task_id='load',
    bash_command='',
    dag=dag
)

server = BashOperator(
    # 클라이언트사이드 페이지열기
    task_id='server',
    bash_command='',
    dag=dag
)

[load, server]

