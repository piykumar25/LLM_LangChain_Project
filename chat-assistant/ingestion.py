import os

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone
from dotenv import load_dotenv
from consts import INDEX_NAME

load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)

def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest", encoding="utf-8")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https://")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    PineconeLangChain.from_documents(documents=documents, embedding=embeddings, index_name=INDEX_NAME)
    print("**** Added to Pinecone vectorstore vectors ****")


if __name__ == '__main__':
    ingest_docs()
