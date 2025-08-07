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
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Document processing and RAG components
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough, Runnable
from langchain.schema import StrOutputParser

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Secure the application with a bearer token
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file.")

# --- Global Context for Models ---
# This dictionary will hold our loaded model, so we don't reload it on every request.
ml_models: Dict[str, Runnable] = {}

# --- FastAPI Lifespan Management (The Fix!) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to load the ML model at startup and clean up on shutdown.
    This ensures the heavy HuggingFace model is loaded only ONCE.
    """
    print("INFO:     Loading HuggingFace embeddings model...")
    # Load the model and store it in the ml_models dictionary
    ml_models["embeddings"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("INFO:     Embeddings model loaded successfully.")
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()
    print("INFO:     ML models cleared.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Query-Retrieval System (Gemini Edition)",
    description="Process PDFs, DOCX, and email documents and answer contextual questions using a RAG pipeline with Google Gemini.",
    version="1.5.0", # Version bump for the fix
    lifespan=lifespan # Use the lifespan manager
)

# --- Pydantic Models for API Data Validation ---
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to the document (PDF, DOCX, EML) to be processed.")
    questions: List[str] = Field(..., min_items=1, description="List of questions to ask about the document.")

class QueryResponse(BaseModel):
    answers: List[str]

# --- Core RAG Logic ---

# --- Document Parsers (Unchanged) ---
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
            if part.get_content_type() == "text/plain":
                text_content += part.get_payload(decode=True).decode(errors='ignore')
            elif part.get_content_type() == "text/html":
                html_content = part.get_payload(decode=True).decode(errors='ignore')
                soup = BeautifulSoup(html_content, "html.parser")
                text_content += soup.get_text()
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            text_content = payload.decode(errors='ignore')
    return text_content

def get_document_text(url: str) -> str:
    try:
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            response = client.get(url)
            response.raise_for_status()
            content = response.content
            content_type = response.headers.get("content-type", "").lower()

            if url.lower().endswith(".pdf") or "application/pdf" in content_type:
                return _parse_pdf(content)
            elif url.lower().endswith(".docx") or "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type:
                return _parse_docx(content)
            elif url.lower().endswith(".eml") or "message/rfc822" in content_type:
                return _parse_email(content)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported document type or failed to detect. URL: {url}, Content-Type: {content_type}")

    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document from URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

def get_text_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250, length_function=len)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks: List[str]):
    """
    Creates a FAISS vector store using the pre-loaded embeddings model.
    """
    try:
        # Use the model loaded at startup instead of reloading it
        embeddings = ml_models.get("embeddings")
        if not embeddings:
            raise HTTPException(status_code=500, detail="Embeddings model not loaded. Please check server logs.")
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {e}")

def get_rag_chain(retriever):
    prompt_template = """
    You are an expert AI policy analyst. Your task is to provide a clear and concise answer to the user's question.
    Base your answer ONLY on the following context extracted from the policy document.
    Do not use any external knowledge. Synthesize the information from the relevant clauses into a direct answer.
    If the answer cannot be found in the provided context, state: "The answer is not explicitly mentioned in the document." and nothing more.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    CONCISE ANSWER:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0, convert_system_message_to_human=True)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_document_and_answer_questions(
    request: QueryRequest,
    authorization: str = Header(None)
):
    if not authorization or authorization.split(" ")[1] != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing Bearer token")

    document_text = get_document_text(request.documents)
    if not document_text:
        raise HTTPException(status_code=500, detail="Could not extract text from the document.")

    text_chunks = get_text_chunks(document_text)
    vector_store = get_vector_store(text_chunks)
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rag_chain = get_rag_chain(retriever)
    
    tasks = [rag_chain.ainvoke(question) for question in request.questions]
    
    try:
        answers = await asyncio.gather(*tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during LLM inference: {e}")

    return QueryResponse(answers=answers)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Intelligent Query-Retrieval System (Gemini Edition) is running."}
