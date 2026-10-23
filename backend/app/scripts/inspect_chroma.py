
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

from app.config.embedding_config import get_embeddings

# Load project-root .env explicitly.
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root.parent / ".env", override=False)

data_dir = project_root.parent / "data"
print("Project root:", project_root.parent)
print("Data directory:", data_dir)

configured_path = os.getenv("CHROMA_PERSIST_DIRECTORY")
print("Configured Chroma path:", configured_path)

# Inspect the intended root data/chroma location.
candidate_paths = [
    data_dir / "chroma",
    project_root / "data" / "chroma",
]

seen = set()

for path in candidate_paths:
    path = path.resolve()

    if path in seen:
        continue
    seen.add(path)

    print(f"\n--- Inspecting: {path} ---")
    print("Exists:", path.exists())

    if not path.exists():
        continue

    print(
        "Files:",
        [p.name for p in path.iterdir()][:20],
    )

    for collection_name in (
        "healthcare_documents",
        "asthma_documents",
        "documents",
    ):
        try:
            store = Chroma(
                collection_name=collection_name,
                persist_directory=str(path),
                embedding_function=get_embeddings(),
            )

            collection = store._collection

            print(
                f"Collection={collection_name!r}, "
                f"count={collection.count()}"
            )

        except Exception as exc:
            print(
                f"Collection={collection_name!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
