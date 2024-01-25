from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings, OllamaEmbeddings
from langchain.chains import RetrievalQA
from pypdf import PdfReader
import streamlit as st
import tempfile
import os
import shutil
from langchain import hub

OLLAMA_URL = "http://172.17.10.68:11434"
# from streamlit.report_thread import get_report_ctx
st.set_page_config(page_title="LLM with RAG", page_icon=":cyclone:")
# -----------Funtions-------------------------------------------------


def _get_session():
    from streamlit.runtime import get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session


user_session = _get_session()
user_session_id = user_session.id
print("user_session", user_session_id)


def save_uploaded_pdf(uploaded_file):
    # Check if a file is uploaded
    if uploaded_file is not None:
        # Create a temporary directory to save the file
        temp_dir = tempfile.mkdtemp()

        # Save the uploaded file to the temporary directory
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Return the path to the saved file
        return file_path


# @st.cache_resource
def get_pdf_chunks(pdf_obj):
    pdf_path = save_uploaded_pdf(pdf_obj)
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    text_splits = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    all_splits = text_splits.split_documents(data)
    return all_splits


# @st.cache_resource
def get_webpage_chunks(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splits = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    all_splits = text_splits.split_documents(data)
    return all_splits


# @st.cache_resource
def get_vectordb_from_chunks(chunks1, chunk2):
    if chunks1 == None:
        all_splits = chunk2
    elif chunk2 == None:
        all_splits = chunks1
    else:
        all_splits = chunks1 + chunk2
    # print("ALL SPLITS", all_splits)

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        # embedding=GPT4AllEmbeddings(),
        embedding=OllamaEmbeddings(
            model="mistral"
        ),  # this is better by far, GPT4AllEmbeddings gives inconsistant query result
        persist_directory="./chroma_db",
        collection_name=user_session_id,
    )

    return vectorstore


def augment_prompt(query: str) -> str:
    # get top 3 results from knowledge base. You can play with k to tune the answers, it also depend on how your knowledge was stored.ie Chuck Size etc
    results = st.session_state.vectorstore.similarity_search(
        query, k=2
    )  # result also return the source documents as metadata
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed source_knowledge to construct a augmented prompt

    # Can try to play around the initial part prompt
    # Using the contexts below, answer the query. OR
    # Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    augmented_prompt = f"""Only use the following Contexts provided bellow to answer the query. If the answers to query is not in the Context, Just say you not there in the Contexts and dont not provide any information regarding it.
    Contexts:
    {source_knowledge}
    Query: {query}"""
    return augmented_prompt


# ---------------------------Declare Session Variable-----------------------------------------
st.title("Chatbot with RAG")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

if "chat_model" not in st.session_state:
    st.session_state.chat_model = ChatOllama(base_url=OLLAMA_URL, model="mistral")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "QA_CHAIN_PROMPT" not in st.session_state:
    st.session_state.prompt_template = hub.pull("rlm/rag-prompt-mistral")


with st.sidebar:
    st.subheader("Your documents")
    pdf_doc = st.file_uploader(
        "Upload your PDF here and click on 'Process'", accept_multiple_files=False
    )
    st.subheader("URL for documentation or article")
    url = st.text_input("URL")
    pdf_chunks = None
    webpage_chunks = None
    vectorstore = None
    if st.button("Process"):
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.session_state.prompt_history = []
        with st.spinner("Processing"):
            # get pdf text
            if pdf_doc:
                # print("pdf_doc", pdf_doc)
                pdf_chunks = get_pdf_chunks(pdf_doc)
            if url:
                webpage_chunks = get_webpage_chunks(url)

            if pdf_doc or webpage_chunks:
                vectorstore = get_vectordb_from_chunks(pdf_chunks, webpage_chunks)
                st.session_state.vectorstore = vectorstore
                print("vectorstore created")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.prompt_history = []


prompt = st.chat_input("Ask question")

# -------------------------------Chat Logic------------------------------------------------------------
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    prompt = HumanMessage(content=augment_prompt(prompt))
    st.session_state.prompt_history.append(prompt)
    res = st.session_state.chat_model(st.session_state.prompt_history)

    st.session_state.messages.append({"role": "assistant", "content": res.content})
    st.chat_message("assistant").markdown(res.content)
    print("########################################")
    print(prompt)
    print("########################################")
    print(res.content)
    print("########################################")
    print(res)


# print("CURRENT", prompt)
# print("url", url)
# print("CURRENT", prompt)
