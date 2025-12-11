"""
Speaker Embedding Cache

LRU cache for zero-shot speaker embeddings to avoid recomputation.
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict
from dataclasses import dataclass, asdict

import torch

logger = logging.getLogger(__name__)


@dataclass
class SpeakerInfo:
    """Information about a registered speaker."""
    speaker_id: str
    prompt_text: str
    prompt_audio_path: Optional[str] = None
    created_at: Optional[str] = None
    

class SpeakerCache:
    """
    Thread-safe LRU cache for speaker embeddings.
    
    Features:
    - LRU eviction policy
    - Disk persistence
    - Thread-safe operations
    """
    
    def __init__(
        self,
        max_size: int = 100,
        persist_path: Optional[str] = None
    ):
        """
        Initialize the speaker cache.
        
        Args:
            max_size: Maximum number of speakers to cache
            persist_path: Directory for persisting speaker info
        """
        self.max_size = max_size
        self.persist_path = Path(persist_path) if persist_path else None
        
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._info: Dict[str, SpeakerInfo] = {}
        self._lock = threading.RLock()
        
        # Create persist directory if needed
        if self.persist_path:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def get(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get speaker embedding from cache.
        
        Args:
            speaker_id: Unique speaker identifier
            
        Returns:
            Speaker embedding dict or None
        """
        with self._lock:
            if speaker_id not in self._cache:
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(speaker_id)
            return self._cache[speaker_id]
    
    def put(
        self,
        speaker_id: str,
        embedding: Dict[str, Any],
        info: Optional[SpeakerInfo] = None
    ) -> None:
        """
        Add speaker embedding to cache.
        
        Args:
            speaker_id: Unique speaker identifier
            embedding: Speaker embedding data
            info: Optional speaker metadata
        """
        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._info:
                    del self._info[oldest_key]
            
            self._cache[speaker_id] = embedding
            if info:
                self._info[speaker_id] = info
    
    def remove(self, speaker_id: str) -> bool:
        """
        Remove speaker from cache.
        
        Args:
            speaker_id: Unique speaker identifier
            
        Returns:
            True if speaker was removed
        """
        with self._lock:
            if speaker_id in self._cache:
                del self._cache[speaker_id]
                if speaker_id in self._info:
                    del self._info[speaker_id]
                return True
            return False
    
    def list_speakers(self) -> list:
        """Get list of all cached speaker IDs."""
        with self._lock:
            return list(self._cache.keys())
    
    def get_info(self, speaker_id: str) -> Optional[SpeakerInfo]:
        """Get speaker metadata."""
        with self._lock:
            return self._info.get(speaker_id)
    
    def all_info(self) -> Dict[str, SpeakerInfo]:
        """Get all speaker metadata."""
        with self._lock:
            return dict(self._info)
    
    def save_to_disk(self) -> None:
        """Persist speaker info to disk."""
        if not self.persist_path:
            return
        
        with self._lock:
            # Save metadata
            info_path = self.persist_path / "speakers.json"
            info_data = {
                sid: asdict(info) 
                for sid, info in self._info.items()
            }
            with open(info_path, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            logger.info(f"Saved {len(info_data)} speakers to {info_path}")
    
    def _load_from_disk(self) -> None:
        """Load speaker info from disk."""
        if not self.persist_path:
            return
        
        info_path = self.persist_path / "speakers.json"
        if not info_path.exists():
            return
        
        try:
            with open(info_path, 'r') as f:
                info_data = json.load(f)
            
            for sid, data in info_data.items():
                self._info[sid] = SpeakerInfo(**data)
            
            logger.info(f"Loaded {len(info_data)} speaker info entries")
        except Exception as e:
            logger.warning(f"Failed to load speaker info: {e}")
    
    def clear(self) -> None:
        """Clear all cached speakers."""
        with self._lock:
            self._cache.clear()
            self._info.clear()
    
    def __len__(self) -> int:
        """Get number of cached speakers."""
        return len(self._cache)
    
    def __contains__(self, speaker_id: str) -> bool:
        """Check if speaker is in cache."""
        return speaker_id in self._cache
