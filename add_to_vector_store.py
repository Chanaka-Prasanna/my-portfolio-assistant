import getpass
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
INDEX_NAME = 'mychat'

if not os.environ.get('GROQ_API_KEY'):
    os.environ['GROQ_API_KEY'] = getpass.getpass("Enter your API key: ")

llm = ChatGroq(model='llama3-8b-8192')

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index(INDEX_NAME)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)


loader = TextLoader(file_path="about_me.txt")
docs = loader.load()



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

document_ids = vector_store.add_documents(documents=all_splits)
