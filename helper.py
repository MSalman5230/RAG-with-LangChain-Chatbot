from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import GPT4AllEmbeddings, OllamaEmbeddings
from langchain.chains import RetrievalQA


# Prompt making function
def augment_prompt(query: str)->str:
    # get top 3 results from knowledge base. You can play with k to tune the answers, it also depend on how your knowledge was stored.ie Chuck Size etc
    results = vectorstore.similarity_search(query, k=2)  # result also return the source documents as metadata
    # get the text from the results
    source_knowledge = "\n".join([x.page_content for x in results])
    # feed source_knowledge to construct a augmented prompt

    # Can try to play around the initial part prompt
    # Using the contexts below, answer the query. OR
    # Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    augmented_prompt = f"""Use the following pieces of information to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Contexts:
    {source_knowledge}
    Query: {query}"""
    return augmented_prompt