# Contract Review RAG API

A FastAPI-based Retrieval-Augmented Generation (RAG) application for automated contract review, clause extraction, risk identification, and comparison with company standards. This project leverages LangChain, Anthropic, HuggingFace, Chroma, and Supabase for a scalable and intelligent legal assistant.

## Features
- Upload and analyze contract PDFs
- Extract and categorize contract clauses
- Identify potential risks
- Compare contracts with company standards
- API key authentication and credit management via Supabase

## Project Structure
- `api_contract_review.py`: Main FastAPI app and core logic
- `requirements.txt`: Python dependencies
- `render.yaml`: Render deployment configuration

## Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ContractReviewRAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root with the following variables:
```
ANTHROPIC_API_KEY=your_anthropic_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### 4. Run the API Locally
```bash
uvicorn api_contract_review:app --reload
```
The API will be available at `http://localhost:8000`.

### 5. Deploy to Render
- Push your code to a GitHub repository.
- Connect your repo to [Render](https://render.com/).
- Render will use `render.yaml` for deployment.

## API Usage
- `POST /analyze_pdf` — Upload a contract PDF and receive extracted clauses and risks (requires API key and credits)
- `GET /` — Health check endpoint

## Environment Variables
- `ANTHROPIC_API_KEY`: API key for Anthropic LLM
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase service key

## License
MIT

---

For questions or support, please open an issue or contact the maintainer.
