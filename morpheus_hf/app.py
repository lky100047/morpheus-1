import streamlit as st
from transformers import pipeline
from morpheus import MorpheusHuggingfaceNLI, MorpheusHuggingfaceQA, MorpheusHuggingfaceSummarization

@st.cache(allow_output_mutation=True)
def load_qa_model():
    model = pipeline("question-answering")
    return model

qa = load_qa_model()
st.title("Ask Questions about your Text")
sentence = st.text_area('Please paste your article :', height=30)
question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")
with st.spinner("Discovering Answers.."):
    if button and sentence:
        answers = qa(question=question, context=sentence)
        st.write(answers['answer'])
