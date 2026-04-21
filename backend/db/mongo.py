import os
from datetime import datetime, timedelta
from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "edge_ai")

_client: MongoClient | None = None

def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client

def get_collection(name: str) -> Collection:
    return get_client()[MONGO_DB][name]

# ── Defect Events ──────────────────────────────────────────────────────────────

def insert_defect_event(doc: dict) -> str:
    """Insert a defect event document. Returns the inserted _id as string."""
    col = get_collection("defect_events")
    result = col.insert_one(doc)
    return str(result.inserted_id)

def get_defect_history(camera_id: str | None = None, limit: int = 10) -> list:
    """Return recent defect events, optionally filtered by camera_id."""
    col = get_collection("defect_events")
    query = {}
    if camera_id:
        query["camera_id"] = camera_id
    return list(col.find(query, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit))

def get_events_paginated(limit: int = 50, skip: int = 0) -> list:
    col = get_collection("defect_events")
    return list(col.find({}, {"_id": 0}).sort("timestamp", DESCENDING).skip(skip).limit(limit))

def get_trend_data(hours: int = 24) -> list:
    """Return hourly defect counts for the last N hours."""
    col = get_collection("defect_events")
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    pipeline = [
        {"$match": {"timestamp": {"$gte": since}}},
        {"$addFields": {"hour": {"$substr": ["$timestamp", 0, 13]}}},
        {"$group": {"_id": "$hour", "count": {"$sum": 1}, "critical": {
            "$sum": {"$cond": [{"$eq": ["$severity", "CRITICAL"]}, 1, 0]}
        }}},
        {"$sort": {"_id": 1}},
    ]
    return list(col.aggregate(pipeline))

def get_system_config() -> dict:
    """Load system config from DB. Falls back to env defaults."""
    col = get_collection("system_config")
    cfg = col.find_one({}, {"_id": 0})
    return cfg or {
        "conf_threshold": float(os.getenv("CONF_THRESHOLD", "0.65")),
        "sample_fps": int(os.getenv("SAMPLE_FPS", "5")),
        "model_version": os.getenv("YOLO_MODEL", "yolov8n.pt"),
        "camera_urls": [os.getenv("CAMERA_URL", "0")],
    }
