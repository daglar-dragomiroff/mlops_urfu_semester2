FROM python:3.10
EXPOSE 8501
WORKDIR /apps
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run question_answerer_streamlit.py