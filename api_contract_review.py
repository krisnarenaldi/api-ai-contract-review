import os
import re
import pandas as pd
from dotenv import load_dotenv
import logging
import io
from PyPDF2 import PdfReader
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.document_loaders import DirectoryLoader

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic  # Changed from OpenAI to Anthropic
# from langchain_community.embeddings import HuggingFaceEmbeddings  # Alternative embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from supabase import create_client

load_dotenv()
class ContractReviewRAG:
    def __init__(self, model_name="claude-3-7-sonnet-20250219", temperature=0, db_directory="contract_db"):
        """
        inisialisasi RAG
        :param model_name: Model LLM yang dipakai
        :param temperature: kreativitas
        :param db_directory: lokasi utk simpan db vektor
        """
        
        # Get API key from environment variable
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
        self.model_name = model_name
        self.temperature = temperature
        self.db_directory = db_directory
        self.llm = ChatAnthropic(model_name=model_name, temperature=temperature, anthropic_api_key=api_key)
        
        # Use HuggingFace embeddings as an alternative to OpenAI embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Store conversation history as messages
        self.messages = []
        
        self.vectorstore = None
        self.qa_chain = None

    def load_contracts(self, contracts_directory):
        """
        Memuat kontrak dari folder dan membuat db vektor
        :param contracts_directory: path ke folder berisi file PDF kontrak
        :return:
        """

        #1. load all file pdf
        loader = DirectoryLoader(
            contracts_directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()

        #2. log jumlah dokumen
        print(f"Loaded {len(documents)} documents pages from {contracts_directory}")

        #3.Membagi dokumen menjadi chunks yang lebih kecil
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n","\n"," ",""]
        )

        chunks = splitter.split_documents(documents)

        print(f"Split into {len(chunks)} chunks for processing")

        #4. Simpan chunk ke db vektor menggunakan Chroma
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_directory
        )

        #5. Simpan DB ke disk - updated method
        # In newer versions, Chroma automatically persists when persist_directory is provided
        # No need to call persist() explicitly
        print(f"Vector db created and saved to {self.db_directory}")

        #6. Setup QA chain
        self._setup_qa_chain()

    def _setup_qa_chain(self):
        """
            Menyiapkan chain QA dengan prompt khusus untuk review kontrak.
        """

        # Template prompt khusus untuk review kontrak
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Anda adalah asisten review kontrak profesional dengan keahlian hukum. "
                      "Gunakan konteks berikut untuk menjawab pertanyaan. "
                      "Jika Anda tidak tahu jawabannya, katakan Anda tidak tahu. JANGAN mencoba membuat jawaban. "
                      "Jika jawabannya tidak ada dalam konteks, katakan bahwa informasi tidak tersedia dalam dokumen."),
            ("human", "Konteks:\n{context}\n\nPertanyaan: {question}"),
        ])

        # Create retrieval chain
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create the QA chain
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Store the chain
        self.qa_chain = qa_chain

    def load_existing_database(self):
        """
        loading vector db
        :return:
        """
        if os.path.exists(self.db_directory):
            self.vectorstore = Chroma(
                persist_directory=self.db_directory,
                embedding_function=self.embeddings
            )
            self._setup_qa_chain()
            print(f"Loaded existing vector database from {self.db_directory}")
        else:
            print(f"No existing db found at {self.db_directory}")

    def analyze_contract(self, query):
        """
        Menganalisan kontrak berdasarkan query
        :param query: pertanyaan tentang kontrak
        :return:
        """
        if not self.qa_chain:
            raise ValueError("Database belum dimuat. Gunakan load_contract() atau load_existing_database() dulu!")

        # Add the user query to message history
        self.messages.append(HumanMessage(content=query))
        
        # Get response from the chain
        response = self.qa_chain.invoke(query)
        
        # Add the AI response to message history
        self.messages.append(AIMessage(content=response))
        
        # Get source documents from retriever using the new invoke method
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)  # Updated from get_relevant_documents to invoke
        
        # Format source information
        sources_info = []
        for doc in docs:
            source_info = {
                "content": doc.page_content[:100] + "...",
                "metadata": doc.metadata
            }
            sources_info.append(source_info)

        return {
            "answer": response,
            "sources": sources_info
        }

    def extract_contract_clauses(self):
        """
        Mengekstrak dan mengkategorikan klausa kontrak utama.
        :return: Dataframe dengan klausa kontrak terorganisir
        """

        # common_clauses = ["Jangka Waktu","Pembayaran","Pengakhiran", "Kerahasiaan",
        #     "Ganti Rugi", "Force Majeure", "Hukum yang Berlaku",
        #     "Penyelesaian Sengketa", "Jaminan", "Pembatasan Tanggung Jawab"]
        common_clauses = ["Jangka Waktu","Pembayaran","Pengakhiran","Ganti Rugi",
                          "Force Majeure", "Penyelesaian Sengketa", "Pembatasan Tanggung Jawab"]

        results = {}

        for clause in common_clauses:
            query = f"Temukan dan kutip bagian kontrak yang berkaitan dengan '{clause}'"
            result = self.analyze_contract(query)
            results[clause] = result["answer"]

        # Konversi ke dataframe untuk tampilan yang lebih baik
        df = pd.DataFrame(list(results.items()),columns=["Klausa","Isi"])
        return df

    def identify_risks(self):
        """
        Mengidentifikasi risiko potensial dalam kontrak
        :return: Dict dengan risiko yang teridentifikasi
        """

        risk_queries = [
            "Adakah klausa yang ambigu atau tidak jelas dalam kontrak ini?",
            "Identifikasi risiko finansial dalam kontrak ini",
            "Apakah ada kewajiban one-sided yang memberatkan salah satu pihak?",
            "Temukan klausa yang mungkin sulit untuk dipatuhi atau diimplementasikan",
            "Apakah ada masalah hukum potensial dalam kontrak ini?"
        ]

        risks = {}

        for query in risk_queries:
            result = self.analyze_contract(query)
            risks[query] = result["answer"]

        return risks

    def compare_with_standard(self,standard_terms_file):
        """
        Membandingkan kontrak dengan standar perusahaan
        :param standard_terms_file: Path ke file yang berisi standar
        :return: Analisa perbedaan dengan standar
        """

        # Memuat syarat standar
        with open(standard_terms_file,"r") as f:
            standard_terms = f.read()

        # Batasi ukuran untuk query
        query = f"""Bandingkan kontrak ini dengan syarat standar berikut dan identifikasi perbedaan signifikan atau penyimpangan: {standard_terms[:1000]}"""

        return  self.analyze_contract(query)


# ---- FastAPI API for PDF upload and analysis ----
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import traceback

# create supabase client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# set up CORS
origins = [
    "https://web-ai-legal-assist.vercel.app",
    "https://web-front-articlesummarizer.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000",
    "*"  # Allow all origins temporarily for testing
]

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is required")

    # Remove any quotes if present
    x_api_key = x_api_key.strip('"')
    print(f"Validating API key: {x_api_key}")  # Debug log

    # Check API key in users table
    try:        

        # Then try our API key query
        response = supabase.table("profiles").select("id").eq("api_key", x_api_key).execute()
        print(f"API key query response: {response.data}")  # Debug log

        if not response.data:
            raise HTTPException(status_code=401, detail="Invalid API Key")

        try:
            user_id = response.data[0]["id"]
            print(f"Successfully got user_id: {user_id}")  # Debug log
        except Exception as e:
            print(f"Error accessing user_id: {str(e)}")  # Debug log
            print(f"Response data structure: {response.data[0]}")  # Debug log
            raise HTTPException(status_code=500, detail=f"Error accessing user data: {str(e)}")

        # Check credits in credits table
        print("Querying credits table...")  # Debug log
        credits_response = supabase.table("credit_contract_reviews").select("credit").eq("user_id", user_id).execute()
        print(f"Credits query response: {credits_response.data}")  # Debug log

        if not credits_response.data:
            raise HTTPException(status_code=401, detail="No credit record found")

        credit = credits_response.data[0]["credit"]
        if credit <= 0:
            raise HTTPException(status_code=401, detail="No credits remaining")

        return {"api_key": x_api_key, "id": user_id, "credits": credits}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using PyPDF2 as a fallback method
    """
    try:
        # First try with PyPDF2
        reader = PdfReader(pdf_path)
        text = ""
        metadata = {"source": pdf_path}
        documents = []
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_metadata = metadata.copy()
                page_metadata["page"] = i + 1
                documents.append(Document(page_content=page_text, metadata=page_metadata))
        
        if not documents:
            logger.warning("PyPDF2 couldn't extract text, PDF might be scanned or have security restrictions")
            return None
            
        return documents
    except Exception as e:
        logger.error(f"Error extracting text with PyPDF2: {str(e)}")
        return None

@app.post("/analyze")
async def analyze_pdf(
    request: Request,
    file_pdf: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    logger.info(f"Received request to analyze PDF: {file_pdf.filename}")
    # Print request information for debugging
    print(f"Request headers: {request.headers}")
    print(f"Content type: {file_pdf.content_type}")
    print(f"Filename: {file_pdf.filename}")
    
    # Validate API key manually instead of using Depends
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is required")
    
    try:
        # Validate API key with Supabase
        logger.info(f"Validating API key: {x_api_key[:5]}...")
        x_api_key = x_api_key.strip('"')
        response = supabase.table("profiles").select("id").eq("api_key", x_api_key).execute()
        
        if not response.data:
            raise HTTPException(status_code=401, detail="Invalid API Key")
        
        user_id = response.data[0]["id"]
        
        # Check credits
        credits_response = supabase.table("credit_contract_reviews").select("credit").eq("user_id", user_id).execute()
        
        if not credits_response.data:
            raise HTTPException(status_code=401, detail="No credit record found")
        
        credit = credits_response.data[0]["credit"]
        if credit <= 0:
            raise HTTPException(status_code=401, detail="No credits remaining")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API key validation error: {str(e)}")
    
    # Validate file type
    if not file_pdf.content_type or "pdf" not in file_pdf.content_type.lower():
        raise HTTPException(status_code=400, detail=f"File must be a PDF. Got content type: {file_pdf.content_type}")
    
    # Save uploaded PDF to a temporary file
    try:
        content = await file_pdf.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    try:
        # Configure PyPDF logging to capture warnings
        pypdf_logger = logging.getLogger("pypdf")
        pypdf_handler = logging.StreamHandler()
        pypdf_logger.addHandler(pypdf_handler)
        pypdf_logger.setLevel(logging.WARNING)
        
        # Load the PDF using PyPDFLoader with error handling
        logger.info(f"Loading PDF from {tmp_path}")
        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            if not documents:
                logger.warning("PyPDFLoader returned empty documents, trying fallback method")
                documents = extract_text_from_pdf(tmp_path)
        except Exception as pdf_error:
            logger.warning(f"Error with PyPDFLoader: {str(pdf_error)}, trying fallback method")
            documents = extract_text_from_pdf(tmp_path)
        
        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. The file may be scanned, corrupted, or password-protected.")
            
        logger.info(f"Successfully loaded {len(documents)} pages from PDF")

        # Split into chunks
        logger.info("Splitting document into chunks")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not split document into chunks. The PDF may not contain extractable text.")
            
        logger.info(f"Split document into {len(chunks)} chunks")

        # Create a new instance for this analysis
        logger.info("Creating vector store")
        contract_rag = ContractReviewRAG()
        contract_rag.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=contract_rag.embeddings,
            persist_directory=None  # No need to persist for single file
        )
        contract_rag._setup_qa_chain()

        # Example analysis: extract main clauses and risks
        logger.info("Extracting contract clauses")
        clauses_df = contract_rag.extract_contract_clauses()
        
        logger.info("Identifying risks")
        risks = contract_rag.identify_risks()

        # Analysis completed successfully, now deduct credit
        try:
            # Deduct 1 credit after successful analysis
            logger.info(f"Deducting credit for user {user_id}")
            supabase.table("credit_contract_reviews").update({"credit": credit - 1}).eq("user_id", user_id).execute()
            logger.info(f"Credit deducted for user {user_id}. Remaining credits: {credit - 1}")
        except Exception as e:
            logger.warning(f"Failed to deduct credit: {str(e)}")
            # Continue anyway since the analysis was successful

        logger.info("Analysis completed successfully")
        return JSONResponse({
            "clauses": clauses_df.to_dict(orient="records"),
            "risks": risks
        })
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
            logger.info(f"Removed temporary file {tmp_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to remove temporary file: {str(cleanup_error)}")

# Add a simple test endpoint
@app.get("/")
async def root():
    return {"message": "Contract Review API is running"}

@app.get("/test")
async def test():
    """Simple endpoint to test if the API is running"""
    return {"status": "ok", "message": "API is running"}

@app.options("/analyze")
async def options_analyze():
    """Handle preflight requests for CORS"""
    return {}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
