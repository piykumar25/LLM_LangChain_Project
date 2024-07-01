import os
from typing import Any, List, Tuple, Dict

from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone
from consts import INDEX_NAME

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_ENVIRONMENT_REGION"])


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docsearch = PineconeLangChain.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True
    # )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="LangChain"), [])
