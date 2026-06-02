import os
import glob
import chromadb
from chromadb.utils import embedding_functions

RAW_DIR = "data/raw"
DB_DIR = "data/chroma_db"

class RAGPipeline:
    def __init__(self, force_reingest=False):
        # We use default SentenceTransformer embeddings for simplicity and offline robustness
        self.client = chromadb.PersistentClient(path=DB_DIR)
        
        if force_reingest:
            try:
                self.client.delete_collection("feynman_corpus")
                print("[RAG] Deleted old collection for re-ingestion.")
            except Exception:
                pass
                
        self.collection = self.client.get_or_create_collection(
            name="feynman_corpus"
        )
        self._initialize_corpus(force=force_reingest)

    def _initialize_corpus(self, force=False):
        # Check if collection is already populated and we're not forcing
        if self.collection.count() > 0 and not force:
            print(f"[RAG] Collection already contains {self.collection.count()} chunks. Skipping ingestion.")
            return
        
        if not os.path.exists(RAW_DIR):
            print(f"[RAG] Raw directory {RAW_DIR} not found. Skipping ingestion.")
            return

        print("[RAG] Ingesting files from raw directory into ChromaDB...")
        
        documents = []
        ids = []
        chunk_id_counter = 0

        # Read all .txt files in RAW_DIR
        for file_path in glob.glob(os.path.join(RAW_DIR, "*.txt")):
            try:
                # We handle UnicodeDecodeError by falling back to errors='ignore'
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                # Simple chunking by double newlines or paragraphs
                chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
                
                for chunk in chunks:
                    documents.append(chunk)
                    ids.append(f"chunk_{chunk_id_counter}")
                    chunk_id_counter += 1
            except Exception as e:
                print(f"[RAG] Error reading {file_path}: {e}")
        
        if not documents:
            print("[RAG] No valid chunks found to ingest.")
            return

        # Add to ChromaDB in batches to avoid limits
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print(f"[RAG] Successfully ingested {len(documents)} chunks into ChromaDB.")

    def retrieve(self, query: str, n_results: int = 3) -> str:
        if self.collection.count() == 0:
            return "No retrieved context available."
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No retrieved context available."
            
        retrieved_chunks = results['documents'][0]
        return "\n\n".join(retrieved_chunks)
