# from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain_teddynote.prompts import load_prompt
from langchain import hub

# 프롬프트 파일 목록 가져오기
import glob


# API KEY 정보로드
# load_dotenv()

st.title("나만의 챗 GPT")

# 맨처음 한번만 실행
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도
    st.session_state["messages"] = []

with st.sidebar:
    clear_button = st.button("대화 초기화")

    prompt_files = glob.glob("prompts/*.yaml")

    selected_prompt = st.selectbox("프롬프트를 선택 해 주세요", prompt_files, index=0)
    task_input = st.text_input("TASK 입력", "")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 대화를 저장하는 함수
def add_message(role: str, message: str):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성
# def create_chain(prompt_type):
def create_chain(prompt_filepath, task=""):
    # prompt | llm | output_parser
    # prompt = load_prompt("prompts/sns.yaml")
    prompt = load_prompt(prompt_filepath)
    if task:
        prompt = prompt.partial(
            task=task
        )  # partial()은 프롬프트 템플릿의 일부 변수를 미리 고정하는 메서드입니다.

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
    chain = create_chain(selected_prompt, task_input)

    # AI의 답변
    ## 한번에 답변
    # ai_answer = chain.invoke({"question": user_input})
    # st.chat_message("assistant").write(ai_answer)

    # 프롬프트 타입에 따라 다른 변수명 사용
    # 요약일떈 ARTICLE 사용, 기본모드일떈 question 사용
    # input_data = (
    #     {"ARTICLE": user_input}
    #     if selected_prompt == "요약"
    #     else {"question": user_input}
    # )

    ## 스트림으로 답변
    # response = chain.stream(input_data)
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
