"""
CosyVoice2 High-Performance TTS Server

FastAPI server with REST and WebSocket endpoints for TTS synthesis.
"""

import os
import io
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosyvoice_tts.engine import CosyVoiceTTSEngine, EngineConfig, get_engine
from cosyvoice_tts.speaker_cache import SpeakerCache, SpeakerInfo
from server.models import (
    TTSSynthesizeRequest, TTSCloneRequest, TTSCrossLingualRequest, TTSInstructRequest,
    SpeakerRegisterRequest, SpeakerListResponse, SpeakerInfo as SpeakerInfoModel,
    HealthResponse, MetricsResponse, ErrorResponse
)
from server.streaming import (
    sync_audio_stream_generator, concat_audio_bytes, 
    WebSocketStreamer, generate_wav_header
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('tts_requests_total', 'Total TTS requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('tts_request_latency_seconds', 'Request latency', ['endpoint'])
AUDIO_DURATION = Histogram('tts_audio_duration_seconds', 'Generated audio duration')

# Global instances
engine: Optional[CosyVoiceTTSEngine] = None
speaker_cache: Optional[SpeakerCache] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global engine, speaker_cache
    
    # Startup
    logger.info("Starting CosyVoice TTS Server...")
    
    # Load configuration
    config = EngineConfig(
        model_path=os.getenv("MODEL_PATH", "pretrained_models/CosyVoice2-0.5B"),
        use_vllm=os.getenv("USE_VLLM", "true").lower() == "true",
        use_jit=os.getenv("USE_JIT", "true").lower() == "true",
        use_trt=os.getenv("USE_TRT", "false").lower() == "true",
        fp16=os.getenv("USE_FP16", "true").lower() == "true",
    )
    
    # Initialize engine
    engine = get_engine(config)
    
    # Initialize speaker cache
    speaker_cache = SpeakerCache(
        max_size=int(os.getenv("SPEAKER_CACHE_SIZE", "100")),
        persist_path=os.getenv("SPEAKER_PERSIST_PATH", "data/speakers")
    )
    
    # Load model (can be deferred for faster startup)
    if os.getenv("LOAD_MODEL_ON_STARTUP", "true").lower() == "true":
        try:
            engine.load()
        except Exception as e:
            logger.error(f"Failed to load model on startup: {e}")
    
    logger.info("Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if speaker_cache:
        speaker_cache.save_to_disk()


# Create FastAPI app
app = FastAPI(
    title="CosyVoice2 TTS Server",
    description="High-performance TTS server with vLLM acceleration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_model_loaded():
    """Ensure the TTS model is loaded."""
    if not engine or not engine.is_loaded:
        if engine:
            engine.load()
        else:
            raise HTTPException(status_code=503, detail="TTS engine not initialized")


# ============== Health & Metrics ==============

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check server health status."""
    gpu_available = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if gpu_available else "cpu"
    
    return HealthResponse(
        status="healthy" if engine and engine.is_loaded else "degraded",
        model_loaded=engine.is_loaded if engine else False,
        gpu_available=gpu_available,
        device=device
    )


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Get Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/v1/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics():
    """Get server metrics."""
    return MetricsResponse(
        total_requests=int(REQUEST_COUNT._value.sum()) if hasattr(REQUEST_COUNT, '_value') else 0,
        total_batches=0,
        avg_batch_size=0.0,
        avg_processing_time_ms=0.0,
        queue_size=0,
        speakers_cached=len(speaker_cache) if speaker_cache else 0
    )


# ============== TTS Endpoints ==============

@app.post("/v1/tts/synthesize", tags=["TTS"])
async def synthesize(request: TTSSynthesizeRequest):
    """
    Synthesize speech from text.
    
    Uses SFT (Supervised Fine-Tuned) mode with a specific speaker voice.
    """
    start_time = time.time()
    ensure_model_loaded()
    
    try:
        # Generate audio
        audio_chunks = []
        for output in engine.synthesize(
            text=request.text,
            speaker_id=request.speaker_id,
            stream=request.stream,
            speed=request.speed
        ):
            if 'tts_speech' in output:
                audio = output['tts_speech']
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
                if audio.ndim > 1:
                    audio = audio.flatten()
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_chunks.append(audio_int16.tobytes())
        
        # Combine and create response
        audio_bytes = concat_audio_bytes(audio_chunks, engine.sample_rate)
        
        REQUEST_COUNT.labels(endpoint='synthesize', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='synthesize').observe(time.time() - start_time)
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='synthesize', status='error').inc()
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts/clone", tags=["TTS"])
async def clone_voice(
    text: str = Form(..., description="Text to synthesize"),
    prompt_text: str = Form(..., description="Transcript of prompt audio"),
    prompt_audio: UploadFile = File(..., description="Reference audio file"),
    speaker_id: Optional[str] = Form(None, description="Cache as this speaker ID"),
    speed: float = Form(1.0, ge=0.5, le=2.0),
    stream: bool = Form(False)
):
    """
    Zero-shot voice cloning.
    
    Clone any voice using a 3-5 second audio sample and its transcript.
    """
    start_time = time.time()
    ensure_model_loaded()
    
    try:
        # Read prompt audio
        audio_bytes = await prompt_audio.read()
        
        # Load audio using engine's load_wav
        from cosyvoice.utils.file_utils import load_wav
        prompt_speech_16k = load_wav(io.BytesIO(audio_bytes), 16000)
        
        # Register speaker if ID provided
        if speaker_id and speaker_cache:
            engine.register_speaker(speaker_id, prompt_text, prompt_speech_16k)
            speaker_cache.put(
                speaker_id,
                {"registered": True},
                SpeakerInfo(speaker_id=speaker_id, prompt_text=prompt_text)
            )
        
        # Generate audio
        audio_chunks = []
        for output in engine.synthesize_zero_shot(
            text=text,
            prompt_text=prompt_text,
            prompt_audio=prompt_speech_16k,
            zero_shot_spk_id=speaker_id or "",
            stream=stream,
            speed=speed
        ):
            if 'tts_speech' in output:
                audio = output['tts_speech']
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
                if audio.ndim > 1:
                    audio = audio.flatten()
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_chunks.append(audio_int16.tobytes())
        
        audio_bytes = concat_audio_bytes(audio_chunks, engine.sample_rate)
        
        REQUEST_COUNT.labels(endpoint='clone', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='clone').observe(time.time() - start_time)
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=cloned.wav"}
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='clone', status='error').inc()
        logger.error(f"Clone error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts/cross-lingual", tags=["TTS"])
async def cross_lingual(
    text: str = Form(..., description="Text to synthesize (can include language tags)"),
    prompt_audio: UploadFile = File(..., description="Reference audio file"),
    speaker_id: Optional[str] = Form(None),
    speed: float = Form(1.0, ge=0.5, le=2.0)
):
    """
    Cross-lingual synthesis.
    
    Generate speech in one language using a voice from another language.
    Use language tags like <|en|>, <|zh|>, <|jp|>, <|ko|>, <|yue|>.
    """
    start_time = time.time()
    ensure_model_loaded()
    
    try:
        audio_bytes = await prompt_audio.read()
        from cosyvoice.utils.file_utils import load_wav
        prompt_speech_16k = load_wav(io.BytesIO(audio_bytes), 16000)
        
        audio_chunks = []
        for output in engine.synthesize_cross_lingual(
            text=text,
            prompt_audio=prompt_speech_16k,
            zero_shot_spk_id=speaker_id or "",
            speed=speed
        ):
            if 'tts_speech' in output:
                audio = output['tts_speech']
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
                if audio.ndim > 1:
                    audio = audio.flatten()
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_chunks.append(audio_int16.tobytes())
        
        audio_bytes = concat_audio_bytes(audio_chunks, engine.sample_rate)
        
        REQUEST_COUNT.labels(endpoint='cross_lingual', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='cross_lingual').observe(time.time() - start_time)
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav"
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='cross_lingual', status='error').inc()
        logger.error(f"Cross-lingual error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts/instruct", tags=["TTS"])
async def instruct_synthesis(
    text: str = Form(..., description="Text to synthesize"),
    instruct_text: str = Form(..., description="Style instruction"),
    prompt_audio: UploadFile = File(..., description="Reference audio file"),
    speaker_id: Optional[str] = Form(None),
    speed: float = Form(1.0, ge=0.5, le=2.0)
):
    """
    Instruction-guided synthesis.
    
    Control speech style with natural language instructions.
    Examples: "speak with excitement", "用四川话说" (speak in Sichuan dialect).
    """
    start_time = time.time()
    ensure_model_loaded()
    
    try:
        audio_bytes = await prompt_audio.read()
        from cosyvoice.utils.file_utils import load_wav
        prompt_speech_16k = load_wav(io.BytesIO(audio_bytes), 16000)
        
        audio_chunks = []
        for output in engine.synthesize_instruct(
            text=text,
            instruct_text=instruct_text,
            prompt_audio=prompt_speech_16k,
            zero_shot_spk_id=speaker_id or "",
            speed=speed
        ):
            if 'tts_speech' in output:
                audio = output['tts_speech']
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
                if audio.ndim > 1:
                    audio = audio.flatten()
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_chunks.append(audio_int16.tobytes())
        
        audio_bytes = concat_audio_bytes(audio_chunks, engine.sample_rate)
        
        REQUEST_COUNT.labels(endpoint='instruct', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='instruct').observe(time.time() - start_time)
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav"
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='instruct', status='error').inc()
        logger.error(f"Instruct error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts/stream", tags=["TTS"])
async def stream_synthesis(
    text: str = Form(...),
    prompt_text: str = Form(None),
    prompt_audio: UploadFile = File(None),
    speaker_id: Optional[str] = Form(None),
    speed: float = Form(1.0, ge=0.5, le=2.0)
):
    """
    Streaming TTS synthesis.
    
    Returns audio chunks as they are generated for low-latency playback.
    """
    ensure_model_loaded()
    
    async def generate():
        try:
            if prompt_audio:
                # Zero-shot mode
                audio_bytes = await prompt_audio.read()
                from cosyvoice.utils.file_utils import load_wav
                prompt_speech_16k = load_wav(io.BytesIO(audio_bytes), 16000)
                
                generator = engine.synthesize_zero_shot(
                    text=text,
                    prompt_text=prompt_text or "",
                    prompt_audio=prompt_speech_16k,
                    zero_shot_spk_id=speaker_id or "",
                    stream=True,
                    speed=speed
                )
            else:
                # SFT mode
                generator = engine.synthesize(
                    text=text,
                    speaker_id=speaker_id,
                    stream=True,
                    speed=speed
                )
            
            # Send WAV header first
            yield generate_wav_header(engine.sample_rate)
            
            for output in generator:
                if 'tts_speech' in output:
                    audio = output['tts_speech']
                    if hasattr(audio, 'numpy'):
                        audio = audio.numpy()
                    if audio.ndim > 1:
                        audio = audio.flatten()
                    audio_int16 = (audio * 32767).astype(np.int16)
                    yield audio_int16.tobytes()
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
    
    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={"Transfer-Encoding": "chunked"}
    )


# ============== WebSocket Endpoint ==============

@app.websocket("/v1/tts/ws")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time TTS streaming.
    
    Send JSON messages with format:
    {
        "text": "Text to synthesize",
        "speaker_id": "optional_speaker_id",
        "speed": 1.0
    }
    
    Receive:
    - {"type": "start", "sample_rate": 22050}
    - Binary audio chunks
    - {"type": "end"}
    """
    await websocket.accept()
    ensure_model_loaded()
    
    streamer = WebSocketStreamer(websocket, engine.sample_rate)
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            text = data.get("text", "")
            speaker_id = data.get("speaker_id")
            speed = data.get("speed", 1.0)
            
            if not text:
                await streamer.send_error("Text is required")
                continue
            
            await streamer.send_start({"text": text})
            
            try:
                for output in engine.synthesize(
                    text=text,
                    speaker_id=speaker_id,
                    stream=True,
                    speed=speed
                ):
                    if 'tts_speech' in output:
                        audio = output['tts_speech']
                        if hasattr(audio, 'numpy'):
                            audio = audio.numpy()
                        if audio.ndim > 1:
                            audio = audio.flatten()
                        audio_int16 = (audio * 32767).astype(np.int16)
                        await streamer.send_audio(audio_int16.tobytes())
                
                await streamer.send_end()
                
            except Exception as e:
                await streamer.send_error(str(e))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ============== Speaker Management ==============

@app.post("/v1/speakers/register", tags=["Speakers"])
async def register_speaker(
    speaker_id: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...)
):
    """
    Register a speaker for reuse.
    
    The speaker embedding is cached for faster synthesis in future requests.
    """
    ensure_model_loaded()
    
    try:
        audio_bytes = await prompt_audio.read()
        from cosyvoice.utils.file_utils import load_wav
        prompt_speech_16k = load_wav(io.BytesIO(audio_bytes), 16000)
        
        success = engine.register_speaker(speaker_id, prompt_text, prompt_speech_16k)
        
        if success and speaker_cache:
            from datetime import datetime
            speaker_cache.put(
                speaker_id,
                {"registered": True},
                SpeakerInfo(
                    speaker_id=speaker_id,
                    prompt_text=prompt_text,
                    created_at=datetime.now().isoformat()
                )
            )
        
        return {"success": success, "speaker_id": speaker_id}
        
    except Exception as e:
        logger.error(f"Speaker registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/speakers", response_model=SpeakerListResponse, tags=["Speakers"])
async def list_speakers():
    """List all available speakers."""
    speakers = []
    
    # Get speakers from engine
    if engine and engine.is_loaded:
        for spk_id in engine.list_speakers():
            info = speaker_cache.get_info(spk_id) if speaker_cache else None
            speakers.append(SpeakerInfoModel(
                speaker_id=spk_id,
                prompt_text=info.prompt_text if info else None,
                created_at=info.created_at if info else None
            ))
    
    return SpeakerListResponse(speakers=speakers, total=len(speakers))


@app.delete("/v1/speakers/{speaker_id}", tags=["Speakers"])
async def delete_speaker(speaker_id: str):
    """Remove a registered speaker."""
    if speaker_cache:
        success = speaker_cache.remove(speaker_id)
        return {"success": success, "speaker_id": speaker_id}
    return {"success": False, "message": "Speaker cache not available"}


# ============== Legacy Endpoints (CosyVoice Compatible) ==============

@app.post("/inference_zero_shot", tags=["Legacy"])
async def legacy_zero_shot(
    tts_text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_wav: UploadFile = File(...)
):
    """Legacy zero-shot endpoint for CosyVoice compatibility."""
    return await clone_voice(
        text=tts_text,
        prompt_text=prompt_text,
        prompt_audio=prompt_wav
    )


@app.post("/inference_cross_lingual", tags=["Legacy"])
async def legacy_cross_lingual(
    tts_text: str = Form(...),
    prompt_wav: UploadFile = File(...)
):
    """Legacy cross-lingual endpoint for CosyVoice compatibility."""
    return await cross_lingual(
        text=tts_text,
        prompt_audio=prompt_wav
    )
