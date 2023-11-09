from typing import Optional
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from llama_index.tools import BaseTool, FunctionTool, QueryEngineTool, ToolMetadata
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, GPTVectorStoreIndex, download_loader
from dotenv import load_dotenv
import openai
from langchain.tools import DuckDuckGoSearchRun, Tool, HumanInputRun
from langchain.chains import LLMMathChain
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv('api_key')
os.environ['PINECONE_API_KEY'] = os.getenv('pinecone_api_key')
os.environ['PINECONE_ENVIRONMENT'] = os.getenv('pinecone_env')

# Create Pinecone vector store and query engine
sampro_webhelp_index_name = 'dnas-sops'
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)
sampro_webhelp_index = pinecone.Index(sampro_webhelp_index_name)
sampro_webhelp_vector_store = PineconeVectorStore(pinecone_index=sampro_webhelp_index)
pinecone_embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
sampro_webhelp_service_context = ServiceContext.from_defaults(embed_model=pinecone_embed_model)
sampro_webhelp_query_index = GPTVectorStoreIndex.from_vector_store(
    vector_store=sampro_webhelp_vector_store,
    service_context=sampro_webhelp_service_context
)
sampro_webhelp_query_engine = sampro_webhelp_query_index.as_query_engine()

# Sampro web help qa tool
sampro_webhelp_tool = QueryEngineTool(
    query_engine=sampro_webhelp_query_engine,
    metadata=ToolMetadata(
        name="sampro_web_help_qa_tool",
        description="Can answer questions related to SamPRO ERP software. Trouble-shooting, user manuals, and general information about SamPRO ERP."
    )
)
sampro_webhelp_lc_tool = sampro_webhelp_tool.to_langchain_tool()

# DNAS QA Tool
dnas_qa_documents = SimpleDirectoryReader("./knowledgeBase/dnas/").load_data()
dnas_qa_index = VectorStoreIndex.from_documents(dnas_qa_documents)
dnas_qa_query_engine = dnas_qa_index.as_query_engine()
dnas_qa_tool = QueryEngineTool(
    query_engine=dnas_qa_query_engine,
    metadata=ToolMetadata(
        name="company_factual_qa_tool",
        description="Information about Day & Nite All Service. Contains information such as contacts, service, history, mission, vision, and DNAS Connect."
    )
)
dnas_qa_lc_tool = dnas_qa_tool.to_langchain_tool()

# SOP QA Tool
sop_documents = SimpleDirectoryReader("./knowledgeBase/sops/").load_data()
sop_index = VectorStoreIndex.from_documents(sop_documents)
sop_query_engine = sop_index.as_query_engine()
sop_tool = QueryEngineTool(
    query_engine=sop_query_engine,
    metadata=ToolMetadata(
        name="standard_of_procedure_qa_tool",
        description="Standard of procedures (SOPs) of Day & Nite All Service. Include SOPs for on/off boarding, management, procurement, hr, IT, etc."
    )
)
sop_lc_tool = sop_tool.to_langchain_tool()

# Policy QA Tool
policy_qa_documents = SimpleDirectoryReader("./knowledgeBase/policies/").load_data()
policy_qa_index = VectorStoreIndex.from_documents(policy_qa_documents)
policy_qa_query_engine = policy_qa_index.as_query_engine()
policy_qa_tool = QueryEngineTool(
    query_engine=policy_qa_query_engine,
    metadata=ToolMetadata(
        name="policy_qa_tool",
        description="Information about Day & Nite All Service policies. Includes office manual, vacation policy, and employee handbook."
    )
)
policy_qa_lc_tool = policy_qa_tool.to_langchain_tool()

# DuckDuckGoSearch
ddg_search = DuckDuckGoSearchRun()
ddg_search_lc_tool = Tool(
    name="Duck Duck Go Search",
    func=ddg_search.run,
    description="Useful for when you need to answer questions about the current events."
)

# Calculator
math = LLMMathChain.from_llm(OpenAI())
math_lc_tool = Tool(
    name="Calculator",
    func=math.run,
    description="Useful for when you need to do math calculations."
)

# Human in the loop
# human_in_the_loop = HumanInputRun()
# human_input_lc_tool = Tool(
#     name="Human in the loop",
#     func=human_in_the_loop.run,
#     description="Useful for when you need more information or clarification."
# )

# Default behavior
def default_response(input_text: Optional[str]):
    return "I cannot answer this question now. More information required."
default_behavior_lc_tool = Tool(
    name="Default response tool",
    func=default_response,
    description="Useful for when you cannot find an appropriate tool to use. Or you don't have enough information to formulate an answer."
)