from typing import TypedDict, List, Optional

class Detection(TypedDict):
    bbox: List[float]       # [x1, y1, x2, y2] in 640x640 space
    class_id: int
    class_name: str
    confidence: float

class AgentState(TypedDict):
    # ── Vision Layer ──
    frame_path: str                    # absolute path to saved PNG
    camera_id: str
    line_id: str
    raw_detections: List[Detection]

    # ── Node 1: Classifier ──
    defect_type: str                   # SCRATCH | CRACK | RUST | DENT | HOLE | DISCOLORATION
    severity: str                      # CRITICAL | MEDIUM | LOW
    confidence: float
    zone: str                          # SURFACE | EDGE | JOINT | CORE | COATING

    # ── Node 2: Root Cause ──
    cause_hypothesis: str
    cause_confidence: float

    # ── Node 3: Action ──
    action: str                        # HALT_LINE | FLAG_QC | LOG_ONLY
    action_rationale: str

    # ── Node 4: Report ──
    report_payload: Optional[dict]
    image_gridfs_id: Optional[str]
    pdf_gridfs_id: Optional[str]

    # ── Metadata ──
    timestamp: str
    event_id: Optional[str]
