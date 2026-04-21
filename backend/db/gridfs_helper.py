import io
from pathlib import Path
from pymongo import MongoClient
import gridfs
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "edge_ai")

_client: MongoClient | None = None
_fs: gridfs.GridFS | None = None

def get_fs() -> gridfs.GridFS:
    global _client, _fs
    if _fs is None:
        _client = MongoClient(MONGO_URI)
        _fs = gridfs.GridFS(_client[MONGO_DB])
    return _fs

def store_image(image_bytes: bytes, filename: str, metadata: dict | None = None) -> str:
    """Store image bytes in GridFS. Returns file_id as string."""
    fs = get_fs()
    file_id = fs.put(image_bytes, filename=filename, metadata=metadata or {})
    return str(file_id)

def store_pdf(pdf_bytes: bytes, filename: str, metadata: dict | None = None) -> str:
    """Store PDF bytes in GridFS. Returns file_id as string."""
    fs = get_fs()
    file_id = fs.put(pdf_bytes, filename=filename,
                     content_type="application/pdf", metadata=metadata or {})
    return str(file_id)

def get_file_bytes(file_id_str: str) -> bytes:
    """Retrieve file bytes from GridFS by string ID."""
    from bson import ObjectId
    fs = get_fs()
    grid_out = fs.get(ObjectId(file_id_str))
    return grid_out.read()

def store_frame_file(frame_path: str, event_id: str) -> str:
    """Read a saved PNG frame and store it in GridFS."""
    path = Path(frame_path)
    if not path.exists():
        return ""
    return store_image(path.read_bytes(), filename=f"{event_id}.png",
                       metadata={"event_id": event_id})
