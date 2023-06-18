import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

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

if __name__=='__main__':
    create_model()
