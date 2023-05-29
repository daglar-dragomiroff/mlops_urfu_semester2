from transformers import pipeline
import streamlit as st


# Создаем и возвращаем пайплайн для поиска ответа на вопрос в указанном тексте.
def get_pipeline():
    qa = pipeline('question-answering')
    return qa


# Данная функция принимает два аргумента:
# context: текст, в котором будет осуществлен поиск ответа на вопрос.
# question: вопрос, на который будет искаться ответ.
# Значения данных аргументов берутся из полей для ввода: CONTEXT и QUESTION соответственно.
def answer_and_score(context, question):
    qa = get_pipeline()
    # Значения context и question получаем из полей ввода (создаются средствами streamlit).
    result = qa(context=context, question=question)

    # Возвращаем кортеж с результатами: первый элемент - это собственно ответ на вопрос,
    # а второй элемент - оценка (score)
    return (result['answer'], result['score'])


if __name__ == '__main__':
    try:
        # Вывод заголовка.
        st.title('ПОИСК В ТЕКСТЕ ОТВЕТА НА ВОПРОС V2.0')
        # Вывод текста с подсказкой.
        st.text('CONTEXT: текст, в котором будет осуществляться поиск ответа (только на английском).\n'
                'QUESTION: вопрос, на который будет осуществляться поиск ответа\n\t в тексте из поля CONTEXT (только на английском).')

        # Поле ввода текста. value - значение по умолчанию.
        context = st.text_input('CONTEXT:', value='My name is Ivan.')
        # Поле ввода вопроса. value - значение по умолчанию.
        question = st.text_input('QUESTION:', value='What is my name?')
        # Кнопка, нажатие на которую запускает процесс поиска.
        result = st.button('ИСКАТЬ ОТВЕТ')
        if result:
            # Моя функция answer_and_score возвращает кортеж, в котором первый элемент
            # собственно ответ на заданный вопрос, а второй - оценка (score) =>
            # в следующей строке делаем распаковку кортежа, и вывод результатов.
            answer, score = answer_and_score(context, question)
            st.text(f'ANSWER={answer}\nSCORE={score}')
    # Если поле CONTEXT и\или QUESTION оставить пустым = ошибка ValueError.
    except ValueError:
        st.text('FILL IN ALL THE INPUT FIELDS! \\ ЗАПОЛНИТЕ ВСЕ ПОЛЯ ВВОДА! (RUS)')
