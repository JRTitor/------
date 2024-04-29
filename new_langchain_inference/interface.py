from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from googletrans import Translator

import streamlit as st
import os
import json

def translate_text(text:str, src_lang:str='en', dest_lang:str='ru') -> str:
    '''
    translate via google translate
    text: str -- string to translate
    src_lang:str -- source language
    dest_lang:str -- target language
    return:str -- return translation from text as string
    '''
    translator = Translator()
    translated_text = translator.translate(text, src=src_lang, dest=dest_lang)
    return translated_text.text

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
st.title('Демо лучшей модели')
input_ru = st.text_input('Напишите сообщение')

## Ollama tinyLlama LLM
llm = Ollama(model='tinyLlama')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


if input_ru:
    # translation
    input_en = translate_text(input_ru, src_lang='ru', dest_lang='en')
    output_en = chain.invoke({'question':input_en})
    output_ru = translate_text(output_en, src_lang='en', dest_lang='ru')
    st.write(output_ru)
