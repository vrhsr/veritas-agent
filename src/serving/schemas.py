"""
Pydantic schemas for the FastAPI serving layer.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="The user's question")
    session_id: Optional[str] = Field(None, description="Session ID for persistent memory")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How does LoRA reduce memory compared to full fine-tuning?",
                "session_id": "user-123",
            }
        }
    }


class QueryResponse(BaseModel):
    session_id: str
    query_type: str = Field(..., description="simple | complex | ambiguous | awaiting_clarification")
    final_answer: str
    cited_sources: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    validation_passed: bool
    retry_count: int
    cost_inr: float
    latency_s: float
    node_latencies: dict
    awaiting_clarification: bool = False


class ClarificationRequest(BaseModel):
    session_id: str
    original_query: str
    answer: str = Field(..., description="User's answer to the clarification question")
