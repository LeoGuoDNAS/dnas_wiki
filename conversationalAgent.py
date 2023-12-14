from langchain.agents import initialize_agent, AgentType, ZeroShotAgent, AgentExecutor, OpenAIFunctionsAgent, ConversationalChatAgent, ConversationalAgent
from langchain.callbacks import StreamlitCallbackHandler, HumanApprovalCallbackHandler, StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory, ChatMessageHistory
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.schema.messages import SystemMessage, HumanMessage, BaseMessage
from langchain.prompts import MessagesPlaceholder
import streamlit as st
from langchain.chat_models import ChatOpenAI
import os
import openai
from dotenv import load_dotenv
from tools import sampro_webhelp_lc_tool, dnas_qa_lc_tool, sop_qa_lc_tool, policy_qa_lc_tool, ddg_search_lc_tool, math_lc_tool, workorder_status_lc_tool
from PIL import Image
from pathlib import Path
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

load_dotenv()
openai.api_key = os.getenv('api_key')
os.environ['OPENAI_API_KEY'] = os.getenv('api_key')
access_code = os.getenv('bot_access_code')

st.set_page_config(page_title="ChatDNAS", page_icon="🌗")
st.image(Image.open(Path("./pics/dnas-logo.png")))
st.title("🌗 ChatDNAS")

# Define session state messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "How can I help you today?"
        },
        # {
        #     "role": "assistant",
        #     "content": "🧠 I can answer questions related to 1. Day & Nite's general information, 2. Standard of Procedure, 3. Day & Nite policies, 4. SamPro WebHelp, 5. workorder status, 6. math calculation and 7. search web"
        # }
    ]

# Define login
def get_access_code():
    if st.session_state['access_code_text_input'] == access_code:
        st.session_state['current_access_code'] = st.session_state['access_code_text_input']
        st.success("Welcome. You can now chat with the bot.") 
    else:
        st.session_state['current_access_code'] = st.session_state['access_code_text_input']
        st.warning("Incorrect, try again.")
with st.expander("🔑 Start here. Login first.", expanded=False):
    st.text_input(
        label="Input Access Code", 
        type="password", 
        help="If you don't know the access code, contact lguo@wearetheone.com", 
        placeholder="If you don't know the access code, contact lguo@wearetheone.com",
        key="access_code_text_input"
    )
    st.button("Submit", on_click=get_access_code)

with st.expander("💡 Learn more about ChatDNAS", expanded=False):
    st.info("""
        🌗 ChatDNAS is a conversational chatbot, designed to empower Day & Nite employees
    """)
    st.info("""
        🧠 ChatDNAS can answer questions related to 
        1. Day & Nite's general information, 
        2. Standard of Procedures, 
        3. Day & Nite policies, 
        4. SamPro WebHelp, 
        5. workorder status, 
        6. math calculation and 
        7. search web for additional information.
    """)

# Define LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)

# Define tools
tools = [
    sampro_webhelp_lc_tool,
    dnas_qa_lc_tool,
    sop_qa_lc_tool,
    policy_qa_lc_tool,
    ddg_search_lc_tool,
    math_lc_tool,
    workorder_status_lc_tool,
]

# Define starter identity prompt
prompt = OpenAIFunctionsAgent.create_prompt(
    SystemMessage(content=(
        "You are ChatDNAS created by Leo Guo at Day & Nite All Service. You are an expert AI assistant at Day & Nite All Service tasked answering questions about the Day & Nite All Service's day-to-day operations, general knowledge, and database inqueries. "
        "You have access to a Day & Nite All Service knowledge bank which you can query but know NOTHING about Day & Nite All Service otherwise. "
        "You should always first query the knowledge bank for information on the concepts in the question. "
        "For example, given the following input question:\n"
        "-----START OF EXAMPLE INPUT QUESTION-----\n"
        "What is Day and Nite and what do they do? \n"
        "-----END OF EXAMPLE INPUT QUESTION-----\n"
        "Your research flow should be:\n"
        "1. Query your dnas_factual_qa_tool for information on 'day and nite general info' to get as much context as you can about it.\n"
        "2. Then, query your standard_of_procedure_qa_tool for information on 'day and nite general info' to get as much context as you can about it.\n"
        "3. Answer the question with the context you have gathered."
        "For another example, given the following input question:\n"
        "-----START OF EXAMPLE INPUT QUESTION-----\n"
        "How do I submit a new workorder request?\n"
        "-----END OF EXAMPLE INPUT QUESTION-----\n"
        "Your research flow should be:\n"
        "1. Query your sampro_webhelp_tool for information on 'submit workorder in sampro' to get as much context as you can about it. \n"
        "2. Answer the question as you now have enough context.\n\n"
        "If you can't find the answer, DO NOT make up an answer. Just say you don't know. "
        "Answer the following question as best you can:")
    ),
    ## Fix this to make memory work.
    # extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")]
)

memory = ConversationTokenBufferMemory(memory_key="chat_history", llm=llm, max_token_limit=2000, human_prefix="user", ai_prefix="assistant")

if st.session_state.messages: 
    for message in st.session_state.messages:
        if message["role"] == "user":
            memory.chat_memory.add_user_message(message["content"])
        else:
            memory.chat_memory.add_ai_message(message["content"])

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

agent_chain = AgentExecutor(
    agent=agent, tools=tools, verbose=True, memory=memory
)

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message("assistant", avatar=Image.open(Path("./pics/bot.png"))).write(msg["content"])

    else:
        st.chat_message("user", avatar="😎").write(msg["content"])

if prompt := st.chat_input(placeholder="Your question here..."):
    if "current_access_code" in st.session_state and st.session_state['current_access_code'] == access_code:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="😎").write(prompt)
        with st.chat_message("assistant", avatar=Image.open(Path("./pics/bot.png"))):
            st.write("🧠 Thinking...")
            st_callback_handler = StreamlitCallbackHandler(st.container())
            try:
                response = agent_chain.run(prompt, callbacks=[st_callback_handler])

            except ValueError as e:
                response = str(e)
                if not response.startswith("Could not parse LLM output: `"):
                    # raise error
                    response = f"{response.replace('Could not parse LLM output:', '')}"
                    response.replace("Could not parse LLM output:", "")
                else:
                    response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    else:
        st.warning("Incorrect or invalid access code")
