import os
import re
import pandas as pd
from dotenv import load_dotenv

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


# if __name__ == "__main__":
#     # Check if API key is set
#     if not os.environ.get("ANTHROPIC_API_KEY"):
#         print("Warning: ANTHROPIC_API_KEY environment variable is not set")
#         print("Please create a .env file with your Anthropic API key or set it directly")
#         print("Example: ANTHROPIC_API_KEY=your-api-key-here")
#     else:
#         print("API key found in environment variables")
#         contract_rag = ContractReviewRAG()
#         print("ContractReviewRAG initialized successfully")
        
#         #2. Muat kontrak
#         contract_rag.load_contracts("./kontrak/")

#         #3. Muat Db
#         contract_rag.load_existing_database()

#         #4. analisa kontrak
#         result = contract_rag.analyze_contract("Apakah terdapat klausul pembatasan tanggung jawab? Jelaskan maksimum tanggung jawab para pihak")
#         print(result["sources"])
#         print(result["answer"])

#         #5. Identifikasi resiko
#         risks = contract_rag.identify_risks()
#         for query, answer in risks.items():
#             print(f"\nQ: {query}")
#             print(f"A: {answer}")

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
    "https://web-ai-legal-assist.vercel.app/",
    "https://web-front-articlesummarizer.vercel.app/",
    "http://localhost:5173",
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

@app.post("/analyze")
async def analyze_pdf(
    request: Request,
    file_pdf: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    # Print request information for debugging
    print(f"Request headers: {request.headers}")
    print(f"Content type: {file_pdf.content_type}")
    print(f"Filename: {file_pdf.filename}")
    
    # Validate API key manually instead of using Depends
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key is required")
    
    try:
        # Validate API key with Supabase
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
        # Load the PDF using PyPDFLoader (single file)
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not split document into chunks")

        # Create a new instance for this analysis
        contract_rag = ContractReviewRAG()
        contract_rag.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=contract_rag.embeddings,
            persist_directory=None  # No need to persist for single file
        )
        contract_rag._setup_qa_chain()

        # Example analysis: extract main clauses and risks
        clauses_df = contract_rag.extract_contract_clauses()
        risks = contract_rag.identify_risks()

        # Analysis completed successfully, now deduct credit
        try:
            # Deduct 1 credit after successful analysis
            supabase.table("credit_contract_reviews").update({"credit": credit - 1}).eq("user_id", user_id).execute()
            print(f"Credit deducted for user {user_id}. Remaining credits: {credit - 1}")
        except Exception as e:
            print(f"Warning: Failed to deduct credit: {str(e)}")
            # Continue anyway since the analysis was successful

        return JSONResponse({
            "clauses": clauses_df.to_dict(orient="records"),
            "risks": risks
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}\n{traceback.format_exc()}")
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

# Add a simple test endpoint
@app.get("/")
async def root():
    return {"message": "Contract Review API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
