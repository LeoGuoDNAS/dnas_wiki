from fastapi import FastAPI, Security, HTTPException
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv
from mangum import Mangum
from starlette import status
import openai

load_dotenv()
openai.api_key = os.getenv('api_key')
client_state = os.getenv('client_state')

app = FastAPI()
handler = Mangum(app)

chatbot_key_1 = os.getenv('chatbot_key_1')

api_keys = [chatbot_key_1]
api_key_header = APIKeyHeader(name="X-API-Key")
def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header in api_keys:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

@app.get("/")
async def root(api_key: str = Security(get_api_key)):
    return {
        "Hello": "mundo",
        "Go to /docs": "for api documentations"
    }

@app.get("/api/v1/chatbot/{query}")
async def ask_question(query: str, api_key: str = Security(get_api_key)):
    # return await chatbot(q=query)
    # return await chatbot(query, [])
    return {}