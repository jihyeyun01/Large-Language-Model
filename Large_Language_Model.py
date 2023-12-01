## 라이브러리
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from PyPDF2 import PdfReader

import streamlit as st
import streamlit_chat as stc
from streamlit_chat import message


from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.callbacks import get_openai_callback

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import re
from translate import Translator



##  function
# 번역
def translate_text(text, target_language='ko'):
    translator = Translator(to_lang=target_language)
    translated = translator.translate(text)
    return translated

# 영어 확인
def has_english_word(text):
    return bool(re.search(r'[a-zA-Z]', text))

# 답변 자르기
def cut_off_response(response, max_response_length):
    if len(response) >= max_response_length:
        cut_off_index = response.rfind('.', 0, max_response_length)
        if cut_off_index != -1:
            response = response[:cut_off_index + 1]
    return response

# 백그라운드 이미지
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# 대화 내용 초기화
def on_btn_click():
    del st.session_state.messages[:]
    del st.session_state.chat_history[:]


def main() :
    st.set_page_config(page_title = 'Ask your PDF')
    
    # 백그라운드 사진
    #add_bg_from_local('background.jpg')  

    #custom_css = """
    #.stApp {
    #    background-color: transparent;
    #    }
    #"""

    #st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black;'>영천시 관광 가이드 챗봇</h1>", unsafe_allow_html=True)

    ## 대화내용 저장 공간
    if "messages" not in st.session_state:
                st.session_state.messages = []

    if "chat_history " not in st.session_state:
                st.session_state.chat_history = []
                
    response_container = st.container()

    container = st.container()

    # 파일 업로드
    # pdf = "영천시관광책자+csv+블로그.pdf"
    pdf = "영천시데이터.pdf"
    
    API_O = st.sidebar.text_input("API-KEY", type="password")

    if API_O:
        # 텍스트 추출
        if pdf is not None :
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages : 
                text += page.extract_text()

            # 청크로 분활
            text_splitter = CharacterTextSplitter(
                separator = "\n",
                chunk_size = 1200,
                chunk_overlap = 200,
                length_function = len
            )
            chunks = text_splitter.split_text(text)

            # 텍스트 임베딩 
            embeddings = OpenAIEmbeddings(openai_api_key = API_O)
            knowledge_base = FAISS.from_texts(chunks, embeddings)

                
            # 대화창 생성
            col3, col4 = st.columns([2,8])
            with col3:
                st.image('yc_chr.png', width = 120)
            with col4:
                with st.form('form', clear_on_submit=True):
                    user_question = st.text_input('영천시 관광에 대해 물어보세요.:sunglasses:','',
                                                placeholder="입력하세요 ...", key="user_question")
                    submitted = st.form_submit_button('전송')

            st.session_state.messages.append(HumanMessage(content = user_question))
            st.session_state.chat_history.append(HumanMessage(content = user_question))

            if submitted : 
                llm = OpenAI(model_name= 'gpt-3.5-turbo-16k', openai_api_key=API_O) # gpt-3.5-turbo-16k 
                    
                    # # 질문 저장
                    # st.session_state.messages.append(HumanMessage(content = user_question))
                    # st.session_state.chat_history.append(HumanMessage(content = user_question))
                    
            
                with get_openai_callback() as cb :  
                    with st.spinner("Thinking...") : 

                        # 메모리에 내용 저장
                        memory = ConversationBufferMemory(memory_key = 'chat_history',
                                                            return_messages = True)
                            
                        # 메모리 및 llm, 텍스트 임베딩 연결
                        chain_memory = ConversationalRetrievalChain.from_llm(llm=llm,
                                                retriever=knowledge_base.as_retriever(),
                                                memory=memory 
                                                )
                        # 답변
                        response = chain_memory.run(question = user_question)

                        # 답변에 영어가 있으면 번역
                        if has_english_word(response):
                            response = translate_text(response, target_language='ko')
                    

                        st.session_state.chat_history.append(AIMessage(content=response))
                        print(cb)
                        
            
                st.session_state.messages.append(AIMessage(content=response))
                


        #custom_css = """
        #.stApp {
        #    background-color: transparent;
        #    }
        #"""

    else:
        st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.stop()

        # Streamlit 앱에 CSS 적용
        # response_container.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)
    
        # 대화 내용 출력
    messages = st.session_state.get('messages', [])
    with response_container:
        con = st.container()
        for i, msg in enumerate(messages[1:]):
            if i == 0:
                con.caption('대화내용')

            if isinstance(msg, HumanMessage):
                message(msg.content, is_user=True, key=str(i) + '_user')
            elif isinstance(msg, AIMessage):
                message(msg.content, is_user=False, key=str(i) + '_ai')

    download_str = []
    for msg in messages[1:]:
        download_str.append(msg.content)
    download_str = '\n'.join(download_str)
    if download_str:
        st.sidebar.download_button('Download', download_str)
    
    

    st.sidebar.button("New Chat", on_click=on_btn_click, type='primary')
    st.sidebar.write("사용법:   \n   질문을 입력하고 '전송'버튼을 누르시면 질문에 대한 답변이 나옵니다.   \n   새로운 대화를 하고 싶거나 질문에 대한 대답에 문제가 발생하시면 위에 있는 'New Chat' 버튼을 눌러 다시 질문해주시면 감사하겠습니다.   \n   대화내용을 다운로드하고 싶으면 질문 후 'New Chat'위에 생성되는 'Download' 버튼을 클릭하시면 txt 파일로 다운로드 가능합니다.")



            

if __name__ == '__main__' :
    main()
