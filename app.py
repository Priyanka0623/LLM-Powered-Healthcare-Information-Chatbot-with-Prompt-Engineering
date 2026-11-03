
import streamlit as st

from src.chatbot import answer_question

st.set_page_config(
    page_title="Healthcare Information Chatbot",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Healthcare Information Chatbot")

st.write(
    "Ask basic questions about hypertension. "
    "Answers are generated only from the provided WHO 2017 manual "
    "and include printed page references."
)

st.info("Educational project based on the WHO 2017 manual.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about hypertension...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding the answer..."):
            answer = answer_question(question)
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

