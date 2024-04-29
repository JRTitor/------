from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
import json

# environment
PATH_ENV = os.path.join(os.getcwd(), 'env.json')
api_data = json.load(open(PATH_ENV))

os.environ['LANGCHAIN_API_KEY'] = api_data['langchain_api']
os.environ['LANGCHAIN_TRACING_V2'] = 'True'

## prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are assistant at delivery service. Answer user. You get 10$ per good response'),
        ('user', 'Question:{question}')
    ]
)

## streamlit
st.title('Langchain Demo delivery service support')
input_txt = st.text_input('write the message to delivery')

## Ollama tinyLlama LLM
llm = Ollama(model='tinyLlama')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


if input_txt:
    st.write(chain.invoke({'question':input_txt}))