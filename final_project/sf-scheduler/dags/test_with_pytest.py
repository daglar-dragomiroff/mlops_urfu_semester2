from airflow.models import DAG
from airflow.operators.python import PythonOperator

import sklearn
from sklearn.metrics import mean_squared_error
import pickle
import pytest
import numpy as np

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

    dataset_1 = np.loadtxt('datasets/dataset_1.txt')
    dataset_2 = np.loadtxt('datasets/dataset_2.txt')
    dataset_3 = np.loadtxt('datasets/dataset_3.txt')
    dataset_noisy = np.loadtxt('datasets/dataset_noisy.txt')
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

# Запускаем pytest
if __name__ == '__main__':
    pytest.main()
    # test_model_with_pytest()