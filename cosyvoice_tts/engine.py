"""
CosyVoice2 High-Performance TTS Engine

Optimized wrapper for CosyVoice2 with vLLM, JIT, and TensorRT acceleration.
"""

import os
import sys
import logging
import threading
from pathlib import Path
from typing import Optional, Generator, Dict, Any, Union
from dataclasses import dataclass

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for TTS Engine."""
    model_path: str = "pretrained_models/CosyVoice2-0.5B"
    use_vllm: bool = True
    use_jit: bool = True
    use_trt: bool = False
    fp16: bool = True
    device: str = "cuda"
    trt_concurrent: int = 1


class CosyVoiceTTSEngine:
    """
    High-performance TTS engine with vLLM acceleration.
    
    Features:
    - vLLM-accelerated LLM inference
    - JIT-compiled flow encoder
    - TensorRT-optimized flow decoder (optional)
    - FP16 inference support
    - Thread-safe initialization
    """
    
    _instance: Optional['CosyVoiceTTSEngine'] = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[EngineConfig] = None):
        """Singleton pattern for engine instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize the TTS engine with configuration."""
        if self._initialized:
            return
            
        self.config = config or EngineConfig()
        self._model = None
        self._sample_rate = 22050
        self._initialized = True
        
    def load(self) -> None:
        """Load the CosyVoice2 model with optimizations."""
        if self._model is not None:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading CosyVoice2 model from {self.config.model_path}")
        logger.info(f"Optimizations: vLLM={self.config.use_vllm}, JIT={self.config.use_jit}, "
                   f"TRT={self.config.use_trt}, FP16={self.config.fp16}")
        
        # Add paths for CosyVoice imports
        # First try: look for CosyVoice repo in the project root
        project_root = Path(__file__).parent.parent
        cosyvoice_repo = project_root / "CosyVoice"
        
        if cosyvoice_repo.exists():
            cosyvoice_path = cosyvoice_repo
            matcha_path = cosyvoice_repo / "third_party" / "Matcha-TTS"
        else:
            # Fallback: relative to model path
            cosyvoice_path = Path(self.config.model_path).parent.parent
            matcha_path = cosyvoice_path / "third_party" / "Matcha-TTS"
        
        if str(cosyvoice_path) not in sys.path:
            sys.path.insert(0, str(cosyvoice_path))
        if matcha_path.exists() and str(matcha_path) not in sys.path:
            sys.path.insert(0, str(matcha_path))
        
        # Register vLLM model if using vLLM
        if self.config.use_vllm:
            try:
                from vllm import ModelRegistry
                from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
                ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
                # Force Qwen2ForCausalLM to use CosyVoice2 implementation
                # This is necessary because the model config likely specifies "Qwen2ForCausalLM"
                ModelRegistry.register_model("Qwen2ForCausalLM", CosyVoice2ForCausalLM)
                logger.info("Registered CosyVoice2ForCausalLM with vLLM (and overrided Qwen2ForCausalLM)")
            except ImportError as e:
                logger.warning(f"vLLM not available, falling back: {e}")
                self.config.use_vllm = False
        
        # Import and initialize CosyVoice2
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
            from cosyvoice.utils.file_utils import load_wav
            
            self._load_wav = load_wav
            # Check for JIT model existence if requested
            if self.config.use_jit:
                jit_suffix = 'fp16' if self.config.fp16 else 'fp32'
                jit_filename = f"flow.encoder.{jit_suffix}.zip"
                jit_path = os.path.join(self.config.model_path, jit_filename)
                
                if not os.path.exists(jit_path):
                    logger.warning(f"JIT model {jit_filename} not found at {jit_path}. "
                                   "Falling back to standard loading (JIT disabled).")
                    self.config.use_jit = False

            self._model = CosyVoice2(
                self.config.model_path,
                load_jit=self.config.use_jit,
                load_trt=self.config.use_trt,
                load_vllm=self.config.use_vllm,
                fp16=self.config.fp16,
                trt_concurrent=self.config.trt_concurrent
            )
            self._sample_rate = self._model.sample_rate
            logger.info(f"Model loaded successfully. Sample rate: {self._sample_rate}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return self._sample_rate
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def synthesize(
        self,
        text: str,
        speaker_id: Optional[str] = None,
        stream: bool = False,
        speed: float = 1.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            speaker_id: Optional speaker ID for voice
            stream: Enable streaming output
            speed: Speech speed multiplier
            
        Yields:
            Dict with 'tts_speech' tensor
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Use SFT inference if speaker_id provided
        if speaker_id:
            for output in self._model.inference_sft(
                text, speaker_id, stream=stream, speed=speed
            ):
                yield output
        else:
            # Default speaker
            available_spks = self._model.list_available_spks()
            if available_spks:
                for output in self._model.inference_sft(
                    text, available_spks[0], stream=stream, speed=speed
                ):
                    yield output
            else:
                raise ValueError("No speakers available. Use zero-shot synthesis.")
    
    def synthesize_zero_shot(
        self,
        text: str,
        prompt_text: str,
        prompt_audio: Union[str, bytes, np.ndarray, torch.Tensor],
        zero_shot_spk_id: str = "",
        stream: bool = False,
        speed: float = 1.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Zero-shot voice cloning synthesis.
        
        Args:
            text: Text to synthesize
            prompt_text: Transcript of prompt audio
            prompt_audio: Reference audio (path, bytes, or tensor)
            zero_shot_spk_id: Cached speaker ID (if already registered)
            stream: Enable streaming output
            speed: Speech speed multiplier
            
        Yields:
            Dict with 'tts_speech' tensor
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Load prompt audio if needed
        if isinstance(prompt_audio, (str, bytes)):
            prompt_speech_16k = self._load_wav(prompt_audio, 16000)
        elif isinstance(prompt_audio, np.ndarray):
            prompt_speech_16k = torch.from_numpy(prompt_audio).float()
        else:
            prompt_speech_16k = prompt_audio
        
        for output in self._model.inference_zero_shot(
            text, prompt_text, prompt_speech_16k,
            zero_shot_spk_id=zero_shot_spk_id,
            stream=stream, speed=speed
        ):
            yield output
    
    def synthesize_cross_lingual(
        self,
        text: str,
        prompt_audio: Union[str, bytes, np.ndarray, torch.Tensor],
        zero_shot_spk_id: str = "",
        stream: bool = False,
        speed: float = 1.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Cross-lingual synthesis (voice in one language, text in another).
        
        Args:
            text: Text to synthesize (can include language tags like <|en|>)
            prompt_audio: Reference audio for voice
            zero_shot_spk_id: Cached speaker ID
            stream: Enable streaming output
            speed: Speech speed multiplier
            
        Yields:
            Dict with 'tts_speech' tensor
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Load prompt audio if needed
        if isinstance(prompt_audio, (str, bytes)):
            prompt_speech_16k = self._load_wav(prompt_audio, 16000)
        elif isinstance(prompt_audio, np.ndarray):
            prompt_speech_16k = torch.from_numpy(prompt_audio).float()
        else:
            prompt_speech_16k = prompt_audio
        
        for output in self._model.inference_cross_lingual(
            text, prompt_speech_16k,
            zero_shot_spk_id=zero_shot_spk_id,
            stream=stream, speed=speed
        ):
            yield output
    
    def synthesize_instruct(
        self,
        text: str,
        instruct_text: str,
        prompt_audio: Union[str, bytes, np.ndarray, torch.Tensor],
        zero_shot_spk_id: str = "",
        stream: bool = False,
        speed: float = 1.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Instruction-guided synthesis (e.g., "speak with excitement").
        
        Args:
            text: Text to synthesize
            instruct_text: Style instruction (e.g., "用四川话说")
            prompt_audio: Reference audio for voice
            zero_shot_spk_id: Cached speaker ID
            stream: Enable streaming output
            speed: Speech speed multiplier
            
        Yields:
            Dict with 'tts_speech' tensor
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Load prompt audio if needed
        if isinstance(prompt_audio, (str, bytes)):
            prompt_speech_16k = self._load_wav(prompt_audio, 16000)
        elif isinstance(prompt_audio, np.ndarray):
            prompt_speech_16k = torch.from_numpy(prompt_audio).float()
        else:
            prompt_speech_16k = prompt_audio
        
        for output in self._model.inference_instruct2(
            text, instruct_text, prompt_speech_16k,
            zero_shot_spk_id=zero_shot_spk_id,
            stream=stream, speed=speed
        ):
            yield output
    
    def register_speaker(
        self,
        speaker_id: str,
        prompt_text: str,
        prompt_audio: Union[str, bytes, np.ndarray, torch.Tensor]
    ) -> bool:
        """
        Register a zero-shot speaker for reuse.
        
        Args:
            speaker_id: Unique identifier for the speaker
            prompt_text: Transcript of prompt audio
            prompt_audio: Reference audio
            
        Returns:
            True if successful
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Load prompt audio if needed
        if isinstance(prompt_audio, (str, bytes)):
            prompt_speech_16k = self._load_wav(prompt_audio, 16000)
        elif isinstance(prompt_audio, np.ndarray):
            prompt_speech_16k = torch.from_numpy(prompt_audio).float()
        else:
            prompt_speech_16k = prompt_audio
        
        return self._model.add_zero_shot_spk(prompt_text, prompt_speech_16k, speaker_id)
    
    def save_speakers(self) -> None:
        """Save registered speakers to disk."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        self._model.save_spkinfo()
        logger.info("Speakers saved")
    
    def list_speakers(self) -> list:
        """Get list of available speaker IDs."""
        if not self.is_loaded:
            return []
        return self._model.list_available_spks()
    
    def audio_to_bytes(
        self,
        audio_tensor: torch.Tensor,
        format: str = "wav"
    ) -> bytes:
        """
        Convert audio tensor to bytes.
        
        Args:
            audio_tensor: Audio tensor from synthesis
            format: Output format (wav, mp3, etc.)
            
        Returns:
            Audio data as bytes
        """
        import io
        import torchaudio
        
        # Ensure correct shape
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor.cpu(), self._sample_rate, format=format)
        buffer.seek(0)
        return buffer.read()
    
    def audio_to_int16(self, audio_tensor: torch.Tensor) -> bytes:
        """
        Convert audio tensor to 16-bit PCM bytes (for streaming).
        
        Args:
            audio_tensor: Audio tensor from synthesis
            
        Returns:
            PCM audio data as bytes
        """
        audio_np = audio_tensor.cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        return audio_int16.tobytes()


# Convenience function
def get_engine(config: Optional[EngineConfig] = None) -> CosyVoiceTTSEngine:
    """Get or create the TTS engine singleton."""
    return CosyVoiceTTSEngine(config)
