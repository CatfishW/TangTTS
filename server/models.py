"""
Pydantic models for API request/response validation.
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported languages."""
    CHINESE = "zh"
    ENGLISH = "en"
    JAPANESE = "jp"
    KOREAN = "ko"
    CANTONESE = "yue"


class TTSSynthesizeRequest(BaseModel):
    """Request for standard TTS synthesis."""
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    speaker_id: Optional[str] = Field(None, description="Speaker ID for voice")
    language: Optional[Language] = Field(None, description="Target language")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    stream: bool = Field(False, description="Enable streaming response")
    format: str = Field("wav", description="Audio output format")


class TTSCloneRequest(BaseModel):
    """Request for zero-shot voice cloning."""
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    prompt_text: str = Field(..., description="Transcript of prompt audio", min_length=1)
    speaker_id: Optional[str] = Field(None, description="Cache as this speaker ID")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    stream: bool = Field(False, description="Enable streaming response")
    format: str = Field("wav", description="Audio output format")


class TTSCrossLingualRequest(BaseModel):
    """Request for cross-lingual synthesis."""
    text: str = Field(..., description="Text to synthesize (can include language tags)")
    speaker_id: Optional[str] = Field(None, description="Cached speaker ID")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    stream: bool = Field(False)
    format: str = Field("wav")


class TTSInstructRequest(BaseModel):
    """Request for instruction-guided synthesis."""
    text: str = Field(..., description="Text to synthesize")
    instruct_text: str = Field(..., description="Style instruction (e.g., 'speak with excitement')")
    speaker_id: Optional[str] = Field(None, description="Cached speaker ID")
    speed: float = Field(1.0, ge=0.5, le=2.0)
    stream: bool = Field(False)
    format: str = Field("wav")


class SpeakerRegisterRequest(BaseModel):
    """Request to register a new speaker."""
    speaker_id: str = Field(..., description="Unique speaker identifier", min_length=1, max_length=64)
    prompt_text: str = Field(..., description="Transcript of prompt audio")


class SpeakerInfo(BaseModel):
    """Information about a registered speaker."""
    speaker_id: str
    prompt_text: Optional[str] = None
    created_at: Optional[str] = None


class SpeakerListResponse(BaseModel):
    """Response containing list of speakers."""
    speakers: List[SpeakerInfo]
    total: int


class TTSResponse(BaseModel):
    """Response for TTS synthesis."""
    success: bool
    message: Optional[str] = None
    audio_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: int = 22050


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    gpu_available: bool
    device: str


class MetricsResponse(BaseModel):
    """Server metrics response."""
    total_requests: int
    total_batches: int
    avg_batch_size: float
    avg_processing_time_ms: float
    queue_size: int
    speakers_cached: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # "start", "audio", "end", "error"
    data: Optional[Dict[str, Any]] = None
    audio: Optional[bytes] = None
    message: Optional[str] = None
