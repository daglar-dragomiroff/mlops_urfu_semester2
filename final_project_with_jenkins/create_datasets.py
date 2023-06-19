import numpy as np
import os 

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

if __name__=='__main__':
    create_datasets()
