from airflow.models import DAG
from airflow.operators.python import PythonOperator

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pickle
import pytest
import numpy as np
import os
import datetime as dt

args = {'owner':'radmil',
        'start_date':dt.datetime(2023,6,10),
        'retries':1,
        'retry_delay':dt.timedelta(minutes=1),
        'depends_on_past':False
        }

def create_datasets():
    # Создадим папку для датасетов
    path_for_datasets = 'datasets'
    isExists = os.path.exists(path_for_datasets)

    if not isExists:
        os.mkdir(path_for_datasets)

    # Генерация dataset_1
    x_1 = np.linspace(0, 10, 100)
    y_1 = x_1 + np.random.rand(100) * 1 - 2
    dataset_1 = np.vstack((x_1, y_1)).T

    # Сохраним dataset_1 google-диск
    np.savetxt(path_for_datasets + '/dataset_1.txt', dataset_1)

    #-----------------------------------------------------------

     # Генерация dataset_2
    x_2 = np.linspace(0, 10, 100)
    y_2 = 5*np.random.rand(100)-1 + 2*x_2
    dataset_2 = np.vstack((x_2, y_2)).T

    # Сохраним dataset_2 google-диск
    np.savetxt(path_for_datasets + '/dataset_2.txt', dataset_2)

    #-----------------------------------------------------------

     # Генерация dataset_3
    x_3 = np.linspace(0, 10, 100)
    y_3 = np.random.rand(100)-2 + 3*x_3
    dataset_3 = np.vstack((x_3, y_3)).T

    # Сохраним dataset_3 google-диск
    np.savetxt(path_for_datasets + '/dataset_3.txt', dataset_3)

    #-----------------------------------------------------------
    
     # Генерация dataset_noisy
    x_noisy = np.arange(100)
    y_noisy = np.random.rand(100) * 10 + 2 + x_noisy

    # Сделаем "сдвиг", который который приведет к очень высокому MSE, который не пройдет pytest
    y_noisy[15:30] = y_noisy[15:30] + 35
    dataset_noisy = np.vstack((x_noisy, y_noisy)).T

    np.savetxt(path_for_datasets + '/dataset_noisy.txt', dataset_noisy)

def create_model():
    dataset_1 = np.loadtxt('datasets/dataset_1.txt')

    # Обучаем модель на dataset_1
    model = LinearRegression()
    X_train = dataset_1[:, 0].reshape(-1, 1)
    y_train = dataset_1[:, 1]
    model.fit(X_train, y_train)

    # Сохранение модели
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

def load_model():
    
    # Загружаем модель + возвращаем ее
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_datasets():
    ''' 
        По умолчанию предполагается что датасеты
        располагаеются в подкаталоге datasets/ в
        текущей рабочей папке + такая структура будет удобнее
        когда будем интегрировать dvc в финальном проекте
        (Предполагаю что эту лабу можно будет взять за основу для финального проекта)
    '''
    path_for_datasets = 'datasets'

    dataset_1 = np.loadtxt(path_for_datasets+'/dataset_1.txt')
    dataset_2 = np.loadtxt(path_for_datasets+'/dataset_2.txt')
    dataset_3 = np.loadtxt(path_for_datasets+'/dataset_3.txt')
    dataset_noisy = np.loadtxt(path_for_datasets+'/dataset_noisy.txt')
    return (dataset_1, dataset_2, dataset_3, dataset_noisy)

def test_model_with_pytest():
    limit = 300
    
    # Загружаем модель
    model = load_model()

    # Загружаем датасеты
    (dataset_1, dataset_2, dataset_3, dataset_noisy) = load_datasets()

    # Проверка на dataset_1
    X_test1 = dataset_1[:, 0].reshape(-1, 1)
    y_test1 = dataset_1[:, 1]
    y_pred1 = model.predict(X_test1)
    MSE_1 = mean_squared_error(y_test1, y_pred1)

    print(f'MSE для dataset 1={MSE_1}')
    assert MSE_1 <= limit, f"ERROR in dataset_1: MSE = {MSE_1}, limit = {limit}"

    # Проверка на dataset_2
    X_test2 = dataset_2[:, 0].reshape(-1, 1)
    y_test2 = dataset_2[:, 1]
    y_pred2 = model.predict(X_test2)
    MSE_2 = mean_squared_error(y_test2, y_pred2)
    print(f"MSE для dataset_2={MSE_2}")
    assert MSE_2 <= limit, f"ERROR in dataset_2: MSE = {MSE_2}, limit = {limit}"

    # Проверка на dataset_3
    X_test3 = dataset_3[:, 0].reshape(-1, 1)
    y_test3 = dataset_3[:, 1]
    y_pred3 = model.predict(X_test3)
    MSE_3 = mean_squared_error(y_test3, y_pred3)
    print(f"MSE для dataset_3={MSE_3}")
    assert MSE_3 <= limit, f"ERROR in dataset_3: MSE = {MSE_3}, limit = {limit}"

    # Проверка на dataset_noisy
    X_test_noisy = dataset_noisy[:, 0].reshape(-1, 1)
    y_test_noisy = dataset_noisy[:, 1]
    y_pred_noisy = model.predict(X_test_noisy)
    MSE_4_noisy = mean_squared_error(y_test_noisy, y_pred_noisy)
    print(f"MSE для dataset_noisy={MSE_4_noisy}")
    assert MSE_4_noisy <= limit, f"ERROR in dataset_noisy: MSE = {MSE_4_noisy}, limit = {limit}"

with DAG(dag_id='create_load_test', default_args=args, schedule=None) as dag:
    create_dataset = PythonOperator(task_id='create_datasets',
                                    python_callable=create_datasets,
                                    dag=dag)
    create_model = PythonOperator(task_id='create_model',
                                    python_callable=create_model,
                                    dag=dag)
    test_model_with_pytest = PythonOperator(task_id='test_model_with_pytest',
                                            python_callable=test_model_with_pytest,
                                            dag=dag)