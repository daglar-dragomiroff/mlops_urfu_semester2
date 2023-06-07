from catboost.datasets import titanic
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import datetime as dt

from airflow.models import DAG
from airflow.operators.python import PythonOperator

args = {'owner':'airflow',
        'start_date':dt.datetime(2020,2,11),
        'retries':1,
        'retry_delay':dt.timedelta(minutes=1),
        'depends_on_past':False
        }

def create_dataset():
    train,test=titanic()

    train['Sex'] = train['Sex'].apply(lambda x: 0 if x=='male' else 1)
    test['Sex'] = test['Sex'].apply(lambda x: 0 if x=='male' else 1)

    train['Age'] = train['Age'].fillna(train.Age.mean())
    test['Age'] = test['Age'].fillna(train.Age.mean())

    train[['Pclass','Sex','Age','SibSp','Parch','Survived']].to_csv('data_train.csv', index=False)
    test[['Pclass','Sex','Age','SibSp','Parch','Survived']].to_csv('data_test.csv', index=False)

def train_model():
    data_train = pd.read_csv('data_train.csv')
    X_train = data_train[['Pclass','Sex','Age','SibSp','Parch']].values
    Y_train = data_train['Survived'].values

    model = LogisticRegression(max_iter=100_000).fit(X_train, Y_train)

    pickle.dump(model, open('model.pkl','wb'))

def make_prediction():
    loaded_model = pickle.load(open('model.pkl','rb'))

    data_test = pd.read_csv('data_test.csv')
    X_test = data_test[['Pclass','Sex','Age','SibSp','Parch']].values

    print(loaded_model.predict(X_test[0:1]))

with DAG(dag_id='test_airflow', default_args=args, schedule=None) as dag:
    create_dataset = PythonOperator(task_id='create_dataset',
                                    python_callable=create_dataset,
                                    dag=dag)
    train_model = PythonOperator(task_id='train_model',
                                 python_callable=train_model,
                                 dag=dag)
    make_prediction = PythonOperator(task_id='make_prediction',
                                     python_callable=make_prediction)