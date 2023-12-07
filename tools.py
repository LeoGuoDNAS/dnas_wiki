from typing import Optional
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from llama_index.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index import ServiceContext, GPTVectorStoreIndex
from dotenv import load_dotenv
import openai
from langchain.tools import DuckDuckGoSearchRun, Tool
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMMathChain
from langchain.llms import OpenAI
import requests
import json
from schemas import statusTranslate, workorder_status_schema
from duckduckgo_search import DDGS
# from langchain.vectorstores import Pinecone

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv('api_key')
os.environ['OPENAI_API_KEY'] = os.getenv('api_key')
os.environ['PINECONE_API_KEY'] = os.getenv('pinecone_api_key')
os.environ['PINECONE_ENVIRONMENT'] = os.getenv('pinecone_env')
os.environ["SERPAPI_API_KEY"] = os.getenv("serp_api_key")
sampro_api_key = os.getenv('sampro_api_key')

# Connect to Pinecone
index_name = "dnas-wiki"
sampro_webhelp_namespace = "sampro-webhelp"
dnas_qa_namespace = 'general-info'
sop_qa_namespace = 'sops'
policy_qa_namespace = 'policies'
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)
pinecone_index = pinecone.Index(index_name)
embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Create Sampro WebHelp Tool
sampro_webhelp_vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    namespace=sampro_webhelp_namespace
)

sampro_webhelp_index = GPTVectorStoreIndex.from_vector_store(
    vector_store=sampro_webhelp_vector_store,
    service_context=service_context
)
sampro_webhelp_query_engine = sampro_webhelp_index.as_query_engine()
sampro_webhelp_tool = QueryEngineTool(
    query_engine=sampro_webhelp_query_engine,
    metadata=ToolMetadata(
        name="sampro_webhelp_tool",
        description="Contains information about Sampro ERP software know-hows, how-tos, manual, specs, etc."
    )
)
sampro_webhelp_lc_tool = sampro_webhelp_tool.to_langchain_tool()

# DNAS QA Tool
dnas_qa_vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    namespace=dnas_qa_namespace
)

dnas_qa_index = GPTVectorStoreIndex.from_vector_store(
    vector_store=dnas_qa_vector_store,
    service_context=service_context
)
dnas_qa_query_engine = dnas_qa_index.as_query_engine()
dnas_qa_tool = QueryEngineTool(
    query_engine=dnas_qa_query_engine,
    metadata=ToolMetadata(
        name="dnas_factual_qa_tool",
        description="Information about Day & Nite All Service. Contains information such as contacts, service, history, mission, vision, and DNAS Connect."
    )
)
dnas_qa_lc_tool = dnas_qa_tool.to_langchain_tool()

# SOP QA Tool
sop_qa_vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    namespace=sop_qa_namespace
)

sop_qa_index = GPTVectorStoreIndex.from_vector_store(
    vector_store=sop_qa_vector_store,
    service_context=service_context
)
sop_qa_query_engine = sop_qa_index.as_query_engine()
sop_qa_tool = QueryEngineTool(
    query_engine=sop_qa_query_engine,
    metadata=ToolMetadata(
        name="standard_of_procedure_qa_tool",
        description="Standard of procedures (SOPs) of Day & Nite All Service. Include SOPs for on/off boarding, management, procurement, hr, IT, etc."
    )
)
sop_qa_lc_tool = sop_qa_tool.to_langchain_tool()

# Policy QA Tool
policy_qa_vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    namespace=policy_qa_namespace
)

policy_qa_index = GPTVectorStoreIndex.from_vector_store(
    vector_store=policy_qa_vector_store,
    service_context=service_context
)
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
# ddg_search = DuckDuckGoSearchRun()
# ddg_search_lc_tool = Tool(
#     name="DuckDuckGoSearch",
#     func=ddg_search.run,
#     description="Useful for when you need to answer questions about the current events."
# )

# SerpAPI Google Search
serpApiSearch = SerpAPIWrapper()
google_search_lc_tool = Tool(
    name="GoogleSearch",
    func=serpApiSearch.run,
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

# Workorder Status Tool
def workorder_status(workorder_id):
    headers = {
        'Authorization': f'api_key {sampro_api_key}',
        'Content-Type': 'application/json'    
    }
    data = {
        'tokenList': [['@WorkorderID@', workorder_id, workorder_id, 'variable']],
        'queryName': 'WorkorderStatusByID'
    }
    response = requests.post(
        'https://sampro.wearetheone.com/DBAnalytics/SAMProAPI.svc/postKPIData', 
        headers=headers, 
        json=data
    )
    text = response.text
    dataFromText = json.loads(json.loads(text))
    if len(dataFromText) != 1:
        return {"RecordNotFound": "Record not exists in our database given your input."}
    else:
        return {
            "Here's the status.": {
                "workorder id": dataFromText[0]['wrkordr_id'],
                "client name": dataFromText[0]['clnt_nme'],
                "client site name": dataFromText[0]['clntste_nme'],
                "client site street number": dataFromText[0]['clntste_stre_nmbr'],
                "work requested (shortened)": dataFromText[0]['wrkordr_wrk_rqstd'].strip()[:100],
                "workorder status": statusTranslate(dataFromText[0]['wrkordr_escltn_stts'].strip().lower())
            }
        }

workorder_status_tool = FunctionTool.from_defaults(
    fn=workorder_status, 
    description="Get workorder status given a workorder id", 
    fn_schema=workorder_status_schema
)
workorder_status_lc_tool = workorder_status_tool.to_langchain_tool()