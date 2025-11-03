# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import (
    exploratory_analysis,
    load_data,
    data_preprocessing,
    build_save_model,
    build_save_dbscan_model,
    load_model_elbow
)

from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'Sanika',
    'start_date': datetime(2025, 1, 15),
    'retries': 1,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'Airflow_Lab1' with the defined default arguments
dag = DAG(
    'Airflow_Lab1_Enhanced',
    default_args=default_args,
    description='Enhanced ML Pipeline with EDA, Agglomerative, and DBSCAN clustering',
    schedule_interval='@daily',  # Run daily (change to '@weekly' or None for manual)
    catchup=False,
)

# Define PythonOperators for each function

# Task 1: Exploratory Data Analysis
exploratory_analysis_task = PythonOperator(
    task_id='exploratory_analysis_task',
    python_callable=exploratory_analysis,
    dag=dag,
)

# Task 2: Load data
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# Task 3: Perform data preprocessing
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task 4: Build and save Agglomerative model
build_save_model_task = PythonOperator(
    task_id='build_agglomerative_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "agglomerative_model.sav"],
    provide_context=True,
    dag=dag,
)

# Task 5: Build and save DBSCAN model (alternative clustering approach)
dbscan_model_task = PythonOperator(
    task_id='dbscan_model_task',
    python_callable=build_save_dbscan_model,
    op_args=[data_preprocessing_task.output, "dbscan_model.sav"],
    dag=dag,
)

# Task 6: Load model and determine optimal clusters
load_model_task = PythonOperator(
    task_id='evaluate_agglomerative_task',
    python_callable=load_model_elbow,
    op_args=["agglomerative_model.sav", build_save_model_task.output],
    dag=dag,
)

# Set task dependencies
# EDA runs first, then data loading, preprocessing, then both clustering models in parallel
exploratory_analysis_task >> load_data_task >> data_preprocessing_task

# After preprocessing, both clustering algorithms run in parallel
data_preprocessing_task >> build_save_model_task >> load_model_task
data_preprocessing_task >> dbscan_model_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()