from langchain.agents import initialize_agent, AgentType, ZeroShotAgent, AgentExecutor, OpenAIFunctionsAgent, ConversationalChatAgent, ConversationalAgent
from langchain.callbacks import StreamlitCallbackHandler, HumanApprovalCallbackHandler, StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.llms import OpenAI
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

st.set_page_config(page_title="ChatDNAS", page_icon="🌗")
st.image(Image.open(Path("./pics/dnas-logo.png")))
st.title("ChatDNAS")

# Define Login
with open('./authConfig.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

authenticator.login('Login', 'sidebar')
if st.session_state["authentication_status"]:
    st.sidebar.warning(body="This is an internal chatbot with access to our own proprietary data. Do NOT share credentials with any third party.", icon="⚠️")
    st.sidebar.info(body="Help me improve! Email Leo lguo@wearetheone.com for suggestions and ideas.", icon="🙌")
    st.sidebar.warning(body="Always logout when you are done.", icon="💭")    
    authenticator.logout(button_name='👋 Logout this session', location='sidebar', key='unique_key')

    # Define LLM
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", streaming=True)

    # Hanlde parsing errors
    def _handle_error(error) -> str:
        return str(error)[:50]
    
    # Define LangChain Agent 2nd Attempt
    tools = [
        sampro_webhelp_lc_tool,
        dnas_qa_lc_tool,
        sop_qa_lc_tool,
        policy_qa_lc_tool,
        ddg_search_lc_tool,
        math_lc_tool,
        workorder_status_lc_tool,
    ]
    prefix = """
        Your name is ChatDNAS. You are created by Leo Guo at Day & Nite All Service.
        Employees at Day & Nite All Service will ask you questions.
        Answering the following questions as best you can. 
        You have access to the following tools:
    """
    # FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:
    # '''
    # Thought: Do I need to use a tool? Yes
    # Action: the action to take, should be one of [{tool_names}]
    # Action Input: the input to the action
    # Observation: the result of the action
    # '''
    # When you have gathered all the information regarding the question, just return the final answer.
    # '''
    # Thought: [Your response generated from the context]
    # '''
    # OR if you do not need to use a tool, just return the following format:
    # Thought: Do I need to use a tool? No
    # Thought: [Your response to the user input in natural language]
    # '''
    # """
    suffix = """Begin!"
    Previous conversation history:
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        # format_instructions=FORMAT_INSTRUCTIONS,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    # prompt = ConversationalAgent.create_prompt(
    #     tools=tools,
    #     prefix=prefix,
    #     suffix=suffix,
    #     ai_prefix="assistant",
    #     human_prefix="Human",
    #     input_variables=["input", "chat_history", "agent_scratchpad"],
    # )
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    
    # agent = ConversationalAgent(llm_chain=llm_chain, memory=memory, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    # Define LangChain Agent
    # agent = initialize_agent(
    #     [
    #         sampro_webhelp_lc_tool,
    #         dnas_qa_lc_tool,
    #         sop_qa_lc_tool,
    #         policy_qa_lc_tool,
    #         google_search_lc_tool,
    #         math_lc_tool,
    #         workorder_status_lc_tool,
    #         # human_in_the_loop
    #     ],
    #     llm=llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     # agent=AgentType.OPENAI_FUNCTIONS,
    #     verbose=True,
    #     handle_paring_errors=_handle_error
    # )
    

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant", 
                "content": "How can I help you today?"
            },
            {
                "role": "assistant",
                "content": "🧠 I can answer questions related to 1. Day & Nite's general information, 2. Standard of Procedure, 3. Day & Nite policies, 4. SamPro WebHelp, 5. workorder status, 6. math calculation and 7. search web"
            }
        ]

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            # st.chat_message("assistant", avatar="🌗").write(msg["content"])
            st.chat_message("assistant", avatar=Image.open(Path("./pics/bot.png"))).write(msg["content"])

        else:
            st.chat_message("user", avatar="😎").write(msg["content"])

    if prompt := st.chat_input(placeholder="Start typing..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="😎").write(prompt)
        # with st.chat_message("assistant", avatar="🌗"):
        with st.chat_message("assistant", avatar=Image.open(Path("./pics/bot.png"))):
            st.write("🧠 Thinking...")
            st_callback_handler = StreamlitCallbackHandler(st.container())
            # st_callback_handler = StreamlitCallbackHandler(st.container())
            # HumanApprovalCallbackHandler()
            # st_callback_handler.on_tool_end()
            # response = agent.run(prompt, callbacks=[st_callback_handler])
            # response = agent_chain.run(prompt, chat_history=st.session_state.messages, callbacks=[st_callback_handler])
            try:
                response = agent_chain.run(prompt, callbacks=[st_callback_handler])
                
                # response = agent_chain.run(input=prompt, chat_history=st.session_state.messages, callbacks=[st_callback_handler])
                # response = agent_chain.run(input=prompt, chat_history=st.session_state.messages, callbacks=[st_callback_handler])
                
                # response = agent.run(input=prompt, chat_history=st.session_state.messages, callbacks=[st_callback_handler])
                # response = agent.run(input=prompt, callbacks=[st_callback_handler])
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

    
elif st.session_state["authentication_status"] is False:
    st.info("Please login first")
    st.sidebar.error('Username/password is incorrect')
    st.sidebar.warning(body='Contact IT if you don\'t have username or password. Do NOT share credentials with third parties.', icon="ℹ️")

elif st.session_state["authentication_status"] is None:
    st.info("Please login first")
    st.sidebar.warning('Please enter your username and password')

# Define LLM to be used
# llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", streaming=True)

# # Hanlde parsing errors
# def _handle_error(error) -> str:
#     return str(error)[:50]

# # Define LangChain Agent
# agent = initialize_agent(
#     [
#         sampro_webhelp_lc_tool,
#         dnas_qa_lc_tool,
#         sop_qa_lc_tool,
#         policy_qa_lc_tool,
#         google_search_lc_tool,
#         math_lc_tool,
#         workorder_status_lc_tool
#     ],
#     llm=llm,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     # agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, # CHAT_ZERO_SHOT_REACT_DESCRIPTION, #ZERO_SHOT_REACT_DESCRIPTION, CONVERSATION..
#     verbose=True,
#     handle_paring_errors=_handle_error
# )

# # st.set_page_config(page_title="Day & Nite Wiki", page_icon="🌗")
# st.image(Image.open(Path("./pics/dnas-logo.png")))
# st.title("Day & Nite Wiki")
# st.info("Help me improve! Email Leo lguo@wearetheone.com for suggestions and ideas.")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [
#         {
#             "role": "assistant", 
#             "content": "How can I help you today?"
#         },
#         {
#             "role": "assistant",
#             "content": "🧠 I can answer questions related to 1. Day & Nite's general information, 2. Standard of Procedure, 3. Day & Nite policies, 4. SamPro WebHelp, 5. workorder status, 6. math calculation and 7. search web"
#         }
#     ]

# for msg in st.session_state.messages:
#     if msg["role"] == "assistant":
#         # st.chat_message("assistant", avatar="🌗").write(msg["content"])
#         st.chat_message("assistant", avatar=Image.open(Path("./pics/bot.png"))).write(msg["content"])

#     else:
#         st.chat_message("user", avatar="😎").write(msg["content"])

# if prompt := st.chat_input(placeholder="Start typing..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user", avatar="😎").write(prompt)
#     # with st.chat_message("assistant", avatar="🌗"):
#     with st.chat_message("assistant", avatar=Image.open(Path("./pics/bot.png"))):
#         st.write("🧠 Thinking...")
#         st_callback_handler = StreamlitCallbackHandler(st.container())
#         # HumanApprovalCallbackHandler()
#         # st_callback_handler.on_tool_end()
#         # response = agent.run(prompt, callbacks=[st_callback_handler])
#         # response = agent.run(prompt, chat_history=st.session_state.messages, callbacks=[st_callback_handler])
#         try:
#             response = agent.run(input=prompt, chat_history=st.session_state.messages, callbacks=[st_callback_handler])
#         except ValueError as e:
#             response = str(e)
#             if not response.startswith("Could not parse LLM output: `"):
#                 # raise e
#                 response = f"{response.replace('Could not parse LLM output:', '')}"
#                 response.replace("Could not parse LLM output:", "")
#             else:
#                 response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")

#         st.session_state.messages.append({"role": "assistant", "content": response})
#         st.write(response)