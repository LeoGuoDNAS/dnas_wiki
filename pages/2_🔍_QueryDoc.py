import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import os
from dotenv import load_dotenv
import shutil
from PIL import Image
from pathlib import Path

load_dotenv()

st.set_page_config(page_title="QueryDoc", page_icon="🔍", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.getenv('api_key')
st.title("🔍 QueryDoc")

# with st.expander("📁 Start here. Upload a file to proceed", expanded=False):
st.info("Upload your documents and talk to them in natural language! File types supported: .pdf, .docx, .txt", icon="💡")
uploaded_file = st.file_uploader("Upload a file to start query", type=["pdf", "docx", "txt"])
index = None
data_directory = "./data/"
if uploaded_file is not None:
    # create data directory
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    # create directory to uploaded file
    path = os.path.join(data_directory, uploaded_file.name)
    
    # write to designated directory
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success("File uploaded successfully!")

    # load file into vectorstore
    with st.spinner(text="Loading and indexing the doc – hang tight!"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo", 
                temperature=0, 
                system_prompt="You are an expert on the uploaded document and your job is to answer questions about the docs. Assume that all questions are related to the docs. Keep your answers technical and based on facts – do not hallucinate features."
            )
        )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)

        if index is None:
            st.warning("Please upload a file to chat with.")
        else:
            if "querydoc_messages" not in st.session_state.keys(): # Initialize the chat messages history
                st.session_state.querydoc_messages = [
                    {"role": "assistant", "content": "Ask me a question about the document!"}
                ]

            if "query_engine" not in st.session_state.keys(): # Initialize the chat engine
                    st.session_state.query_engine = index.as_query_engine()

            if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
                st.session_state.querydoc_messages.append({"role": "user", "content": prompt})

            for message in st.session_state.querydoc_messages: # Display the prior chat messages
                # with st.chat_message(message["role"], avatar=Image.open(Path("./pics/bot.png"))):
                #     st.write(message["content"])
                if message["role"] == "assistant":
                    with st.chat_message("assistant", avatar=Image.open(Path("./pics/bot.png"))):
                        st.write(message["content"])
                else:
                    with st.chat_message("user", avatar="😎"):
                        st.write(message["content"])

            # If last message is not from assistant, generate a new response
            if "querydoc_messages" in st.session_state.keys():
                if st.session_state.querydoc_messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = st.session_state.query_engine.query(prompt)
                            st.write(response.response)
                            message = {"role": "assistant", "content": response.response}
                            st.session_state.querydoc_messages.append(message) # Add response to message history
else:
    if os.path.exists(data_directory):
        shutil.rmtree(data_directory)
        st.session_state.querydoc_messages = [{"role": "assistant", "content": "Ask me a question about the document!"}]
        st.warning("No data is available. Upload a file to proceed. ☝️")
        st.session_state.clear()
        


# import streamlit as st
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone, VectorStore, Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import UnstructuredFileLoader
# from langchain.document_loaders.image import UnstructuredImageLoader
# from langchain.document_loaders import ImageCaptionLoader
# from langchain.docstore.document import Document
# from llama_index import VectorStoreIndex
# import os
# # import pytube
# import openai
# from dotenv import load_dotenv

# load_dotenv()

# # Chat UI title
# st.header("🔍 QueryDoc")
# st.info("Upload your documents and talk to them in natural language! File types supported: .pdf, .csv, .docx, .txt", icon="💡")

# # File uploader in the sidebar on the left
# with st.sidebar:
#     # Input for OpenAI API Key
#     openai_api_key = os.getenv('api_key')

#     # Check if OpenAI API Key is provided
#     if not openai_api_key:
#         st.info("Please add your OpenAI API key to continue.")
#         st.stop()

#     # Set OPENAI_API_KEY as an environment variable
#     os.environ["OPENAI_API_KEY"] = openai_api_key

# # Initialize ChatOpenAI model
# llm = ChatOpenAI(temperature=0, max_tokens=16000, model_name="gpt-3.5-turbo-16k", streaming=True)

# # Load version history from the text file
# # def load_version_history():
# #     with open("version_history.txt", "r") as file:
# #         return file.read()

# # Sidebar section for uploading files and providing a YouTube URL
# with st.sidebar:
#     uploaded_files = st.file_uploader("Please upload your files.", help="Supported file extensions: .pdf, .csv, .docx, .txt", accept_multiple_files=True, type=["pdf", "csv", "docx", "txt"])
#     # youtube_url = st.text_input("YouTube URL")

#     # Create an expander for the version history in the sidebar
#     # with st.sidebar.expander("**Version History**", expanded=False):
#     #     st.write(load_version_history())

#     st.info("Please refresh the browser to reset the session", icon="🚨")

# # Check if files are uploaded or YouTube URL is provided
# if uploaded_files:
#     # Print the number of files uploaded or YouTube URL provided to the console
#     st.write(f"Number of files uploaded: {len(uploaded_files)}")

#     # Load the data and perform preprocessing only if it hasn't been loaded before
#     if "processed_data" not in st.session_state:
#         # Load the data from uploaded files
#         documents = []

#         if uploaded_files:
#             for uploaded_file in uploaded_files:
#                 # Get the full file path of the uploaded file
#                 file_path = os.path.join(os.getcwd(), uploaded_file.name)

#                 # Save the uploaded file to disk
#                 with open(file_path, "wb") as f:
#                     f.write(uploaded_file.getvalue())

#                 # Check if the file is an image
#                 if file_path.endswith((".png", ".jpg")):
#                     # Use ImageCaptionLoader to load the image file
#                     image_loader = ImageCaptionLoader(path_images=[file_path])

#                     # Load image captions
#                     image_documents = image_loader.load()

#                     # Append the Langchain documents to the documents list
#                     documents.extend(image_documents)
                    
#                 elif file_path.endswith((".pdf", ".docx", ".txt")):
#                     # Use UnstructuredFileLoader to load the PDF/DOCX/TXT file
#                     loader = UnstructuredFileLoader(file_path)
#                     loaded_documents = loader.load()

#                     # Extend the main documents list with the loaded documents
#                     documents.extend(loaded_documents)

#         # Load the YouTube audio stream if URL is provided
#         # if youtube_url:
#         #     youtube_video = pytube.YouTube(youtube_url)
#         #     streams = youtube_video.streams.filter(only_audio=True)
#         #     stream = streams.first()
#         #     stream.download(filename="youtube_audio.mp4")
#         #     # Set the API key for Whisper
#         #     openai.api_key = openai_api_key
#         #     with open("youtube_audio.mp4", "rb") as audio_file:
#         #         transcript = openai.Audio.transcribe("whisper-1", audio_file)
#         #     youtube_text = transcript['text']

#         #     # Create a Langchain document instance for the transcribed text
#         #     youtube_document = Document(page_content=youtube_text, metadata={})
#         #     documents.append(youtube_document)

#         # Chunk the data, create embeddings, and save in vectorstore
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        
#         document_chunks = text_splitter.split_documents(documents)

#         embeddings = OpenAIEmbeddings()
#         vectorstore = VectorStoreIndex.from_documents(documents=document_chunks)
        
#         # vectorstore = Pinecone.from_documents(document_chunks, embeddings)

#         # Store the processed data in session state for reuse
#         st.session_state.processed_data = {
#             "document_chunks": document_chunks,
#             "vectorstore": vectorstore,
#         }

#     else:
#         # If the processed data is already available, retrieve it from session state
#         document_chunks = st.session_state.processed_data["document_chunks"]
#         vectorstore = st.session_state.processed_data["vectorstore"]

#     # Initialize Langchain's QA Chain with the vectorstore
#     qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

#     # Initialize chat history
#     if "querydoc_messages" not in st.session_state:
#         st.session_state.querydoc_messages = []

#     # Display chat messages from history on app rerun
#     for message in st.session_state.querydoc_messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Accept user input
#     if prompt := st.chat_input("Ask your questions?"):
#         st.session_state.querydoc_messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Query the assistant using the latest chat history
#         result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.querydoc_messages]})

#         # Display assistant response in chat message container
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             full_response = ""
#             full_response = result["answer"]
#             message_placeholder.markdown(full_response + "|")
#         message_placeholder.markdown(full_response)    
#         print(full_response)
#         st.session_state.querydoc_messages.append({"role": "assistant", "content": full_response})

# else:
#     st.warning("Please upload a file to proceed.", icon="👈")