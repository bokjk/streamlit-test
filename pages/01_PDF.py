import streamlit as st

from dotenv import load_dotenv

from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain_teddynote.prompts import load_prompt
from langchain import hub

# 프롬프트 파일 목록 가져오기
import glob

import os

# API KEY 정보로드
load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


st.title("PDF 기반 QA")

# 맨처음 한번만 실행
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도
    st.session_state["messages"] = []

with st.sidebar:
    # 초기화 버튼
    clear_button = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    selected_prompt = "prompts/pdf-rag.yaml"


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 대화를 저장하는 함수
def add_message(role: str, message: str):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시에 저장 (시간이 오래 걸리는 작업을 처리 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다... ")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)


# 파일이 업로드 되었을때
if uploaded_file:
    embed_file(uploaded_file)


# 체인 생성
def create_chain(prompt_filepath):
    # prompt | llm | output_parser
    # prompt = load_prompt("prompts/sns.yaml")
    prompt = load_prompt(prompt_filepath)

    # GPT
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser
    return chain


# 초기화 버튼이 눌리면
if clear_button:
    # st.write("대화 초기화 버튼 클릭")
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)

    # chain을 생성
    chain = create_chain(selected_prompt)

    ## 스트림으로 답변
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍출력한다.
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화를 저장
    add_message(role="user", message=user_input)
    add_message(role="assistant", message=ai_answer)
