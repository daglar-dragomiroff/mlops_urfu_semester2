FROM apache/airflow:2.5.3

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

COPY --chown=airflow:root dags/ /opt/airflow/dags/
