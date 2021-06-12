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
    'iter',
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
    bash_command='python ${WORKING_DIRECTORY}/db_to_file.py',
    dag=dag,
)

retrain = BashOperator(
    # 재학습
    task_id='retrain',
    bash_command='python ${WORKING_DIRECTORY}/dkt/train.py \
                    --model_dir ${WORKING_DIRECTORY}/models \
                    --asset_dir ${WORKING_DIRECTORY}/asset \
                    --data_dir ${WORKING_DIRECTORY} --file_name data.csv',
    dag=dag,
)

packing = BashOperator(
    # 패키징
    task_id='pack',
    bash_command='',
    dag=dag
)

drop_reload = BashOperator(
    # 기존의 컨테이너 내리기
    task_id='drop',
    bash_command='',
    dag=dag
)

dataload >> retrain >> packing >> drop_reload
