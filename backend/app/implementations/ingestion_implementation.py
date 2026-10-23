from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.config import settings
from app.implementations.vector_store import get_vector_store

class IngestionImplementation:
    def __init__(self):
        self.root = settings.resolved_documents_dir
        self.root.mkdir(parents=True, exist_ok=True)
    def _paths(self, requested):
        if not requested: paths = list(self.root.glob("*.pdf")) + list(self.root.glob("*.txt"))
        else:
            paths = []
            for item in requested:
                p = (self.root / item).resolve()
                if self.root.resolve() not in p.parents: raise ValueError("Only paths under DOCUMENTS_DIR are allowed")
                if not p.is_file(): raise ValueError(f"File not found: {item}")
                paths.append(p)
        return sorted(paths)
    def ingest(self, file_paths, rebuild=False):
        paths = self._paths(file_paths)
        if not paths: raise ValueError(f"No PDF/TXT files found in {self.root}")
        docs, details = [], []
        for path in paths:
            loaded = PyPDFLoader(str(path)).load() if path.suffix.lower()==".pdf" else TextLoader(str(path), encoding="utf-8").load()
            for i, d in enumerate(loaded):
                d.metadata.update({"source":path.name,"document_id":path.name,"page":d.metadata.get("page",i)})
            docs.extend(loaded); details.append({"file":path.name,"pages_or_documents":len(loaded)})
        splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap, add_start_index=True,
            separators=["\n\n","\n",". "," ",""])
        chunks = splitter.split_documents(docs)
        for i, c in enumerate(chunks): c.metadata["chunk_id"] = f'{c.metadata.get("source")}:{c.metadata.get("page")}:{i}'
        store = get_vector_store()
        if rebuild:
            store.delete_collection()
            store = get_vector_store(force_recreate=True)
        ids = [f'{c.metadata.get("source")}:{c.metadata.get("page")}:{i}' for i,c in enumerate(chunks)]
        store.add_documents(chunks, ids=ids)
        return {"status":"completed","files_processed":len(paths),"pages_loaded":len(docs),
                "chunks_created":len(chunks),"collection":settings.chroma_collection,"details":details}
