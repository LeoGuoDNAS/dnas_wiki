from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain.chat_models import ChatOpenAI
import os
import openai
from dotenv import load_dotenv
from tools import dnas_qa_lc_tool, sop_lc_tool, sampro_webhelp_lc_tool, policy_qa_lc_tool, ddg_search_lc_tool, math_lc_tool, default_behavior_lc_tool

load_dotenv()
openai.api_key = os.getenv('api_key')
os.environ['OPENAI_API_KEY'] = os.getenv('api_key')

# Define LLM to be used
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)

# Define LangChain Agent
agent = initialize_agent(
    [
        dnas_qa_lc_tool,
        sop_lc_tool,
        sampro_webhelp_lc_tool,
        policy_qa_lc_tool,
        ddg_search_lc_tool,
        math_lc_tool,
        default_behavior_lc_tool
    ],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_paring_errors=True
)

# agent.run("What is the vacation policy? How many days do we get off? And what are they?")

st.set_page_config(page_title="Day & Nite Wiki", page_icon="🤖")
st.title("☀️🌑🤖 Day & Nite Wiki")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "How can I help you today?"
        },
        {
            "role": "assistant", 
            "content": "I can answer questions about 1. Day & Nite's company general information, 2. Standard of Procedure, 3. Day & Nite policeis, and 4. SamPro WebHelp."
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Start typing..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st.write("☀️🌑🤖 Thinking...")
        st_callback_handler = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(prompt, callbacks=[st_callback_handler])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
    