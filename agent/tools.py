from pathlib import Path
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

OLLAMA_URL = "http://192.168.1.158:11434"
MODEL = "mistral"

BASE_DATA_PATH = Path("data")
COC_PDF = BASE_DATA_PATH / "code_of_conduct.pdf"
TAX_PDF = BASE_DATA_PATH / "tax_guidelines.pdf"
COC_DB_DIR = BASE_DATA_PATH / "coc_chroma"
TAX_DB_DIR = BASE_DATA_PATH / "tax_chroma"

embedder = OllamaEmbeddings(model=MODEL, base_url=OLLAMA_URL)


def _load_vectorstore(pdf_path: Path, db_path: Path) -> Chroma:
    if db_path.exists():
        return Chroma(persist_directory=str(db_path), embedding_function=embedder)

    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=str(db_path)
    )
    return store


_coc_store = None
_tax_store = None


def fetch_from_coc(question: str, k: int = 4) -> str:
    global _coc_store
    if _coc_store is None:
        _coc_store = _load_vectorstore(COC_PDF, COC_DB_DIR)
    results = _coc_store.similarity_search(question, k=k)
    return "\n\n".join([r.page_content for r in results])


def fetch_from_tax(question: str, k: int = 4) -> str:
    global _tax_store
    if _tax_store is None:
        _tax_store = _load_vectorstore(TAX_PDF, TAX_DB_DIR)
    results = _tax_store.similarity_search(question, k=k)
    return "\n\n".join([r.page_content for r in results])
