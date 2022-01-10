import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from morpheus import MorpheusHuggingfaceNLI, MorpheusHuggingfaceQA, MorpheusHuggingfaceSummarization

model_name = "deepset/roberta-base-squad2"

@st.cache(allow_output_mutation=True)
def load_qa_model():
    model = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return model

qa = load_qa_model()
test_morph_qa = MorpheusHuggingfaceQA('deepset/roberta-base-squad2')

st.title("Ask Questions about your Text")
sentence = st.text_area('Please paste your article :', height=30)
question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")
with st.spinner("Discovering Answers.."):
    if button and sentence:
        answers = qa(question=question, context=sentence)
        st.write(answers['answer'], answers['Test'])
        
