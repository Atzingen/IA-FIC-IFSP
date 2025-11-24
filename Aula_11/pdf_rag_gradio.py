"""Simple Gradio chat interface for asking questions about OsElementos-Euclides.pdf."""

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


PDF_NAME = "OsElementos-Euclides.pdf"


def ensure_api_key() -> str:
    """Load .env and return the OpenAI API key, raising if missing."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in Aula_11/.env before running.")
    return api_key


@lru_cache(maxsize=1)
def build_rag_chain():
    """Build and memoize the RAG chain for the target PDF."""
    ensure_api_key()
    pdf_path = Path(__file__).with_name(PDF_NAME)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Load and split the PDF into overlapping chunks for retrieval.
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Você é um assistente que responde em português usando exclusivamente o contexto fornecido.",
        ),
        (
            "human",
            "Contexto:\n{context}\n\nPergunta:\n{question}",
        ),
    ])

    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def answer_question(message: str, history: list[list[str]]):
    """Gradio-compatible handler that feeds user input into the RAG pipeline."""
    rag_chain = build_rag_chain()
    response = rag_chain.invoke(message)
    return response


def main():
    ensure_api_key()
    demo = gr.ChatInterface(
        fn=answer_question,
        title="RAG sobre Os Elementos",
        description=(
            "Faça perguntas sobre o conteúdo do PDF 'Os Elementos' e receba respostas baseadas"
            " nas passagens recuperadas do texto."
        ),
        examples=[
            ["Resuma as ideias do início do livro."],
            ["Quais postulados são apresentados?"],
        ],
    )
    demo.launch()


if __name__ == "__main__":
    main()
