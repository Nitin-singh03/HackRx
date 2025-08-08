# filename: server.py
import os
import asyncio
import httpx
import io
import email
from email.policy import default
from bs4 import BeautifulSoup
import docx

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv

# Document processing and RAG components
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

# --- Configuration ---
load_dotenv()

API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# --- In-Memory Cache ---
VECTOR_STORE_CACHE: Dict[str, FAISS] = {}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Query-Retrieval System (Gemini Edition)",
    description="An efficient, cached RAG pipeline for processing documents.",
    version="2.0.1" # Version bumped for prompt fix
)

# --- Pydantic Models (Unchanged) ---
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to the document (PDF, DOCX, EML) to be processed.")
    questions: List[str] = Field(..., min_items=1, description="List of questions to ask about the document.")

class QueryResponse(BaseModel):
    answers: List[str]

# --- Core Logic Functions (Unchanged) ---
def _parse_pdf(content: bytes) -> str:
    with fitz.open(stream=content, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def _parse_docx(content: bytes) -> str:
    with io.BytesIO(content) as stream:
        doc = docx.Document(stream)
        return "\n".join([para.text for para in doc.paragraphs])

def _parse_email(content: bytes) -> str:
    msg = email.message_from_bytes(content, policy=default)
    text_content = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                text_content += part.get_payload(decode=True).decode(errors='ignore')
            elif content_type == "text/html":
                html_content = part.get_payload(decode=True).decode(errors='ignore')
                soup = BeautifulSoup(html_content, "html.parser")
                text_content += soup.get_text()
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text_content = payload.decode(errors='ignore')
    return text_content

async def get_document_text(url: str) -> str:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.content
            content_type = response.headers.get("content-type", "").lower()
            if url.lower().endswith(".pdf") or "application/pdf" in content_type:
                return _parse_pdf(content)
            elif url.lower().endswith(".docx"):
                return _parse_docx(content)
            elif url.lower().endswith(".eml") or "message/rfc822" in content_type:
                return _parse_email(content)
            else:
                raise HTTPException(status_code=400, detail="Unsupported document type.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document from URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

def get_text_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250, length_function=len)
    return text_splitter.split_text(text)

# ★★★ THIS FUNCTION IS NOW CORRECTED ★★★
def get_rag_chain(retriever):
    """Builds the RAG chain with the full, correct prompt."""
    prompt_template = """
    You are a highly skilled AI assistant specialized in analyzing insurance policy documents.
    Your task is to answer the user's question based *exclusively* on the provided context.

    Follow these rules strictly:
    1.  Your answer must be a single, complete, and professionally worded sentence.
    2.  The answer must be derived *only* from the information within the 'CONTEXT' block. Do not use any external knowledge.
    3.  Incorporate key details like numbers, durations (e.g., 36 months, 2 years), and percentages directly into your answer.
    4.  If the context does not contain the information needed to answer the question, you MUST respond with the exact phrase: "The answer could not be found in the provided document."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        convert_system_message_to_human=True
    )
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Main API Endpoint with Caching Logic ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_document_and_answer_questions(
    request: QueryRequest,
    authorization: str = Header(None)
):
    if not API_BEARER_TOKEN or not authorization or authorization.split(" ")[1] != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

    doc_url = request.documents

    if doc_url in VECTOR_STORE_CACHE:
        print(f"CACHE HIT: Reusing vector store for {doc_url}")
        vector_store = VECTOR_STORE_CACHE[doc_url]
    else:
        print(f"CACHE MISS: Building new vector store for {doc_url}")
        document_text = await get_document_text(doc_url)
        if not document_text:
            raise HTTPException(status_code=500, detail="Could not extract text from the document.")

        text_chunks = get_text_chunks(document_text)
        if not text_chunks:
            raise HTTPException(status_code=500, detail="Document is empty or text could not be chunked.")

        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            vector_store = await FAISS.afrom_texts(texts=text_chunks, embedding=embeddings)
            VECTOR_STORE_CACHE[doc_url] = vector_store
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create vector store: {e}")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    rag_chain = get_rag_chain(retriever)
    
    tasks = [rag_chain.ainvoke(question) for question in request.questions]
    
    try:
        answers = await asyncio.gather(*tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during LLM inference: {e}")

    return QueryResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "ok", "message": f"Intelligent Query-Retrieval System is running. Items in cache: {len(VECTOR_STORE_CACHE)}"}