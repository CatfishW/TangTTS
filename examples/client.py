#!/usr/bin/env python3
"""
CosyVoice TTS Client Example

Demonstrates how to use the TTS API from Python.
"""

import os
import asyncio
import aiohttp
import httpx
from pathlib import Path


class CosyVoiceTTSClient:
    """
    Async client for CosyVoice TTS server.
    
    Usage:
        async with CosyVoiceTTSClient("http://localhost:8000") as client:
            audio = await client.synthesize("Hello world")
            with open("output.wav", "wb") as f:
                f.write(audio)
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=60.0)
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    async def health_check(self) -> dict:
        """Check server health."""
        response = await self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def list_speakers(self) -> list:
        """Get list of available speakers."""
        response = await self._client.get(f"{self.base_url}/v1/speakers")
        response.raise_for_status()
        return response.json()["speakers"]
    
    async def synthesize(
        self,
        text: str,
        speaker_id: str = None,
        speed: float = 1.0
    ) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            speaker_id: Optional speaker ID
            speed: Speech speed multiplier (0.5-2.0)
            
        Returns:
            WAV audio data as bytes
        """
        response = await self._client.post(
            f"{self.base_url}/v1/tts/synthesize",
            json={
                "text": text,
                "speaker_id": speaker_id,
                "speed": speed
            }
        )
        response.raise_for_status()
        return response.content
    
    async def clone_voice(
        self,
        text: str,
        prompt_text: str,
        prompt_audio: bytes,
        speaker_id: str = None,
        speed: float = 1.0
    ) -> bytes:
        """
        Zero-shot voice cloning.
        
        Args:
            text: Text to synthesize
            prompt_text: Transcript of prompt audio
            prompt_audio: Reference audio as bytes
            speaker_id: Optional ID to cache the voice
            speed: Speech speed multiplier
            
        Returns:
            WAV audio data as bytes
        """
        files = {
            "prompt_audio": ("prompt.wav", prompt_audio, "audio/wav")
        }
        data = {
            "text": text,
            "prompt_text": prompt_text,
            "speed": str(speed)
        }
        if speaker_id:
            data["speaker_id"] = speaker_id
        
        response = await self._client.post(
            f"{self.base_url}/v1/tts/clone",
            data=data,
            files=files
        )
        response.raise_for_status()
        return response.content
    
    async def cross_lingual(
        self,
        text: str,
        prompt_audio: bytes,
        speed: float = 1.0
    ) -> bytes:
        """
        Cross-lingual synthesis.
        
        Args:
            text: Text to synthesize (can include language tags like <|en|>)
            prompt_audio: Reference audio as bytes
            speed: Speech speed multiplier
            
        Returns:
            WAV audio data as bytes
        """
        files = {
            "prompt_audio": ("prompt.wav", prompt_audio, "audio/wav")
        }
        data = {
            "text": text,
            "speed": str(speed)
        }
        
        response = await self._client.post(
            f"{self.base_url}/v1/tts/cross-lingual",
            data=data,
            files=files
        )
        response.raise_for_status()
        return response.content
    
    async def register_speaker(
        self,
        speaker_id: str,
        prompt_text: str,
        prompt_audio: bytes
    ) -> dict:
        """
        Register a speaker for reuse.
        
        Args:
            speaker_id: Unique identifier for the speaker
            prompt_text: Transcript of prompt audio
            prompt_audio: Reference audio as bytes
            
        Returns:
            Registration result
        """
        files = {
            "prompt_audio": ("prompt.wav", prompt_audio, "audio/wav")
        }
        data = {
            "speaker_id": speaker_id,
            "prompt_text": prompt_text
        }
        
        response = await self._client.post(
            f"{self.base_url}/v1/speakers/register",
            data=data,
            files=files
        )
        response.raise_for_status()
        return response.json()
    
    async def stream_synthesize(
        self,
        text: str,
        speaker_id: str = None,
        speed: float = 1.0
    ):
        """
        Stream synthesized audio.
        
        Args:
            text: Text to synthesize
            speaker_id: Optional speaker ID
            speed: Speech speed multiplier
            
        Yields:
            Audio chunks as bytes
        """
        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/tts/stream",
            data={
                "text": text,
                "speaker_id": speaker_id or "",
                "speed": str(speed)
            }
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


# Synchronous wrapper for convenience
class SyncClient:
    """Synchronous wrapper for the async client."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def _request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        with httpx.Client(timeout=60.0) as client:
            response = getattr(client, method)(
                f"{self.base_url}{endpoint}",
                **kwargs
            )
            response.raise_for_status()
            return response
    
    def synthesize(self, text: str, speaker_id: str = None, speed: float = 1.0) -> bytes:
        """Synthesize speech from text."""
        response = self._request(
            "post",
            "/v1/tts/synthesize",
            json={"text": text, "speaker_id": speaker_id, "speed": speed}
        )
        return response.content
    
    def clone_voice(
        self,
        text: str,
        prompt_text: str,
        prompt_audio_path: str,
        speed: float = 1.0
    ) -> bytes:
        """Clone a voice from an audio file."""
        with open(prompt_audio_path, "rb") as f:
            prompt_audio = f.read()
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/v1/tts/clone",
                data={"text": text, "prompt_text": prompt_text, "speed": str(speed)},
                files={"prompt_audio": ("prompt.wav", prompt_audio, "audio/wav")}
            )
            response.raise_for_status()
            return response.content


# Example usage
async def main():
    """Example usage of the TTS client."""
    
    print("CosyVoice TTS Client Example")
    print("=" * 40)
    
    async with CosyVoiceTTSClient("http://localhost:8000") as client:
        # Check health
        try:
            health = await client.health_check()
            print(f"Server status: {health['status']}")
            print(f"Model loaded: {health['model_loaded']}")
            print(f"GPU: {health['device']}")
        except Exception as e:
            print(f"Server not available: {e}")
            return
        
        # List speakers
        speakers = await client.list_speakers()
        print(f"\nAvailable speakers: {len(speakers)}")
        
        # Synthesize text
        print("\nSynthesizing 'Hello, this is a test.'...")
        try:
            audio = await client.synthesize(
                "Hello, this is a test of the CosyVoice text to speech system.",
                speed=1.0
            )
            
            output_path = "output_example.wav"
            with open(output_path, "wb") as f:
                f.write(audio)
            print(f"Audio saved to: {output_path}")
            print(f"Audio size: {len(audio)} bytes")
            
        except Exception as e:
            print(f"Synthesis failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
