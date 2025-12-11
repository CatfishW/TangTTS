"""
Streaming utilities for real-time audio delivery.
"""

import asyncio
import logging
import struct
from typing import AsyncGenerator, Generator, Dict, Any
import io

logger = logging.getLogger(__name__)


def generate_wav_header(
    sample_rate: int = 22050,
    num_channels: int = 1,
    bits_per_sample: int = 16
) -> bytes:
    """
    Generate a WAV header for streaming.
    
    Uses 0xFFFFFFFF for data size to indicate streaming.
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        0xFFFFFFFF,  # File size (unknown for streaming)
        b'WAVE',
        b'fmt ',
        16,  # Subchunk1Size
        1,   # AudioFormat (PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        0xFFFFFFFF  # Data size (unknown for streaming)
    )
    return header


async def audio_stream_generator(
    model_output_generator: Generator[Dict[str, Any], None, None],
    sample_rate: int = 22050,
    include_header: bool = True
) -> AsyncGenerator[bytes, None]:
    """
    Convert TTS model output to streaming audio bytes.
    
    Args:
        model_output_generator: Generator from TTS model
        sample_rate: Audio sample rate
        include_header: Include WAV header at start
        
    Yields:
        Audio data chunks as bytes
    """
    import numpy as np
    
    if include_header:
        yield generate_wav_header(sample_rate)
    
    for output in model_output_generator:
        if 'tts_speech' in output:
            audio = output['tts_speech']
            
            # Convert to numpy if tensor
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            
            # Flatten if multi-dimensional
            if audio.ndim > 1:
                audio = audio.flatten()
            
            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            yield audio_int16.tobytes()
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)


def sync_audio_stream_generator(
    model_output_generator: Generator[Dict[str, Any], None, None],
    sample_rate: int = 22050,
    include_header: bool = True
) -> Generator[bytes, None, None]:
    """
    Synchronous version of audio stream generator.
    
    Args:
        model_output_generator: Generator from TTS model
        sample_rate: Audio sample rate
        include_header: Include WAV header at start
        
    Yields:
        Audio data chunks as bytes
    """
    import numpy as np
    
    if include_header:
        yield generate_wav_header(sample_rate)
    
    for output in model_output_generator:
        if 'tts_speech' in output:
            audio = output['tts_speech']
            
            # Convert to numpy if tensor
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            
            # Flatten if multi-dimensional
            if audio.ndim > 1:
                audio = audio.flatten()
            
            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            yield audio_int16.tobytes()


def concat_audio_bytes(chunks: list, sample_rate: int = 22050) -> bytes:
    """
    Concatenate audio chunks and add proper WAV header.
    
    Args:
        chunks: List of audio byte chunks (without headers)
        sample_rate: Audio sample rate
        
    Returns:
        Complete WAV file as bytes
    """
    import numpy as np
    
    # Concatenate all chunks
    audio_data = b''.join(chunks)
    data_size = len(audio_data)
    file_size = data_size + 36
    
    # Create proper header with known sizes
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        file_size,
        b'WAVE',
        b'fmt ',
        16,  # Subchunk1Size
        1,   # AudioFormat (PCM)
        1,   # Channels
        sample_rate,
        sample_rate * 2,  # ByteRate
        2,   # BlockAlign
        16,  # BitsPerSample
        b'data',
        data_size
    )
    
    return header + audio_data


class WebSocketStreamer:
    """
    WebSocket-based audio streamer.
    
    Handles chunked delivery over WebSocket connection.
    """
    
    def __init__(
        self,
        websocket,
        sample_rate: int = 22050,
        chunk_size_ms: int = 100
    ):
        """
        Initialize the WebSocket streamer.
        
        Args:
            websocket: WebSocket connection
            sample_rate: Audio sample rate
            chunk_size_ms: Target chunk size in milliseconds
        """
        self.websocket = websocket
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_size_ms / 1000) * 2  # bytes
        self._buffer = io.BytesIO()
    
    async def send_start(self, metadata: Dict[str, Any] = None) -> None:
        """Send stream start message."""
        import json
        await self.websocket.send_text(json.dumps({
            "type": "start",
            "sample_rate": self.sample_rate,
            "metadata": metadata or {}
        }))
    
    async def send_audio(self, audio_bytes: bytes) -> None:
        """Send audio chunk."""
        self._buffer.write(audio_bytes)
        
        # Send when we have enough data
        while self._buffer.tell() >= self.chunk_size:
            self._buffer.seek(0)
            chunk = self._buffer.read(self.chunk_size)
            remaining = self._buffer.read()
            self._buffer = io.BytesIO()
            self._buffer.write(remaining)
            
            await self.websocket.send_bytes(chunk)
    
    async def send_end(self, metadata: Dict[str, Any] = None) -> None:
        """Send stream end message."""
        import json
        
        # Flush remaining buffer
        if self._buffer.tell() > 0:
            self._buffer.seek(0)
            await self.websocket.send_bytes(self._buffer.read())
        
        await self.websocket.send_text(json.dumps({
            "type": "end",
            "metadata": metadata or {}
        }))
    
    async def send_error(self, error: str) -> None:
        """Send error message."""
        import json
        await self.websocket.send_text(json.dumps({
            "type": "error",
            "message": error
        }))
