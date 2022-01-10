import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from morpheus import MorpheusHuggingfaceNLI, MorpheusHuggingfaceQA, MorpheusHuggingfaceSummarization

model_name = "deepset/roberta-base-squad2"

@st.cache(allow_output_mutation=True)
def load_qa_model():
    model = pipeline('question-answering', model=model_name, tokenizer=model_name)
    return model

qa = load_qa_model()



st.title("Ask Questions about your Text")
sentence = st.text_area('Please paste your article :', height=30)
question = st.text_input("Questions from this article?")
button = st.button("Get me Answers")
with st.spinner("Discovering Answers.."):
    if button and sentence:
        answers = qa(question=question, context=sentence)
        st.write(answers['answer'], answers['Test'])
        
        test_morph_qa = MorpheusHuggingfaceQA('deepset/roberta-base-squad2')
        context = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
        q_dict = {"question": "In what country is Normandy located?", "id": "56ddde6b9a695914005b9628", "answers": [{"text": "France", "answer_start": 159}, {"text": "France", "answer_start": 159}, {"text": "France", "answer_start": 159}, {"text": "France", "answer_start": 159}], "is_impossible": False}

        print(test_morph_qa.morph(q_dict, context))
