from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler, HumanApprovalCallbackHandler
import streamlit as st
from langchain.chat_models import ChatOpenAI
import os
import openai
from dotenv import load_dotenv
from tools import sampro_webhelp_lc_tool, dnas_qa_lc_tool, sop_qa_lc_tool, policy_qa_lc_tool, ddg_search_lc_tool, math_lc_tool, workorder_status_lc_tool
from PIL import Image
from pathlib import Path

load_dotenv()
openai.api_key = os.getenv('api_key')
os.environ['OPENAI_API_KEY'] = os.getenv('api_key')


# Define LLM to be used
llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", streaming=True)

# Hanlde parsing errors
def _handle_error(error) -> str:
    return str(error)[:50]

# Define LangChain Agent
agent = initialize_agent(
    [
        sampro_webhelp_lc_tool,
        dnas_qa_lc_tool,
        sop_qa_lc_tool,
        policy_qa_lc_tool,
        ddg_search_lc_tool,
        math_lc_tool,
        workorder_status_lc_tool
    ],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # CHAT_ZERO_SHOT_REACT_DESCRIPTION, #ZERO_SHOT_REACT_DESCRIPTION, CONVERSATION..
    verbose=True,
    handle_paring_errors=_handle_error
)

st.set_page_config(page_title="Day & Nite Wiki", page_icon="🌗")
st.title("🌗 Day & Nite Wiki")

st.info("I am an AI Chatbot capable of answering questions related to 1. Day & Nite's general information, 2. Standard of Procedure, 3. Day & Nite policies, 4. SamPro WebHelp, 5. workorder status, 6. math calculation and 7. search web for additional info.")
st.info("Help me improve! Email lguo@wearetheone.com to suggestions and ideas.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "How can I help you today?"
        }
    ]

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message("assistant", avatar="🌗").write(msg["content"])
    else:
        st.chat_message("user", avatar="🤔").write(msg["content"])

if prompt := st.chat_input(placeholder="Start typing..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🤔").write(prompt)
    with st.chat_message("assistant", avatar="🌗"):
        st.write("🧠 Thinking...")
        st_callback_handler = StreamlitCallbackHandler(st.container())
        # HumanApprovalCallbackHandler()
        # st_callback_handler.on_tool_end()
        # response = agent.run(prompt, callbacks=[st_callback_handler])
        # response = agent.run(prompt, chat_history=st.session_state.messages, callbacks=[st_callback_handler])
        try:
            response = agent.run(input=prompt, chat_history=st.session_state.messages, callbacks=[st_callback_handler])
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                # raise e
                response = f"{response.replace('Could not parse LLM output:', '')}"
                response.replace("Could not parse LLM output:", "")
            else:
                response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)