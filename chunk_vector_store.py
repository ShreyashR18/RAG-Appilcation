from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma;
from langchain_community.embeddings import FastEmbedEmbeddings;
import chromadb


class ChunkVectorStore:

  def __init__(self) -> None:
    pass

  def split_into_chunks(self, file_path: str):
    doc = PyPDFLoader(file_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    chunks = text_splitter.split_documents(doc)
    chunks = filter_complex_metadata(chunks)

    return chunks

  def store_to_vector_database(self, chunks):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    # Initialize ChromaDB with persistent storage
    persist_directory = "./chroma_db"  # Set a valid directory for ChromaDB storage
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=FastEmbedEmbeddings(),
        persist_directory=persist_directory
    )
    
    return vector_store  # Return the LangChain Chroma object

