from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

# Define paths
BASE_DATA_PATH = Path("data")
COC_PDF = BASE_DATA_PATH / "code_of_conduct.pdf"
TAX_PDF = BASE_DATA_PATH / "tax_guidelines.pdf"
COC_DB_DIR = BASE_DATA_PATH / "coc_faiss"
TAX_DB_DIR = BASE_DATA_PATH / "tax_faiss"

# Initialize the embedding model
embedder = OllamaEmbeddings(model="mistral", base_url="http://192.168.1.158:11434")

# Function to load or create a FAISS vector store
def _load_vectorstore(pdf_path: Path, db_path: Path) -> FAISS:
    if db_path.exists():
        return FAISS.load_local(str(db_path), embedder, allow_dangerous_deserialization=True)

    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    store = FAISS.from_documents(chunks, embedder)
    store.save_local(str(db_path))
    return store

# Initialize global variables
_coc_store = None
_tax_store = None

# Function to fetch from Code of Conduct vector store
def fetch_from_coc(question: str, k: int = 4) -> str:
    global _coc_store
    if _coc_store is None:
        _coc_store = _load_vectorstore(COC_PDF, COC_DB_DIR)
    results = _coc_store.similarity_search(question, k=k)
    return "\n\n".join([r.page_content for r in results])

# Function to fetch from Tax Guidelines vector store
def fetch_from_tax(question: str, k: int = 4) -> str:
    global _tax_store
    if _tax_store is None:
        _tax_store = _load_vectorstore(TAX_PDF, TAX_DB_DIR)
    results = _tax_store.similarity_search(question, k=k)
    return "\n\n".join([r.page_content for r in results])
