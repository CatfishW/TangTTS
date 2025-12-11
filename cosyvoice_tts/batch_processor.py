"""
Request Batch Processor

Intelligent batching for TTS requests to maximize throughput.
"""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass
class TTSRequest:
    """A TTS synthesis request."""
    request_id: str
    text: str
    mode: str  # "sft", "zero_shot", "cross_lingual", "instruct"
    params: Dict[str, Any] = field(default_factory=dict)
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None


@dataclass
class TTSResult:
    """Result of a TTS request."""
    request_id: str
    audio_data: bytes
    sample_rate: int
    duration_seconds: float
    processing_time_ms: float


class BatchProcessor:
    """
    Asynchronous batch processor for TTS requests.
    
    Groups similar requests and processes them together for better
    GPU utilization and throughput.
    """
    
    def __init__(
        self,
        process_fn: Callable[[List[TTSRequest]], Awaitable[List[TTSResult]]],
        max_batch_size: int = 8,
        timeout_ms: int = 100,
        enabled: bool = True
    ):
        """
        Initialize the batch processor.
        
        Args:
            process_fn: Async function to process a batch of requests
            max_batch_size: Maximum requests per batch
            timeout_ms: Max wait time before processing partial batch
            enabled: Whether batching is enabled
        """
        self.process_fn = process_fn
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.enabled = enabled
        
        self._queue: asyncio.PriorityQueue = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_requests = 0
        self._total_batches = 0
        self._total_processing_time = 0.0
    
    async def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            return
        
        self._queue = asyncio.PriorityQueue()
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Batch processor started")
    
    async def stop(self) -> None:
        """Stop the batch processor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Batch processor stopped")
    
    async def submit(self, request: TTSRequest) -> TTSResult:
        """
        Submit a request for processing.
        
        Args:
            request: The TTS request
            
        Returns:
            The TTS result
        """
        if not self.enabled:
            # Process immediately without batching
            results = await self.process_fn([request])
            return results[0]
        
        if not self._running:
            await self.start()
        
        # Create future for result
        request.future = asyncio.get_event_loop().create_future()
        
        # Add to priority queue (lower priority value = higher priority)
        priority = -request.priority.value
        await self._queue.put((priority, request.created_at, request))
        
        self._total_requests += 1
        
        # Wait for result
        return await request.future
    
    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[TTSRequest]:
        """Collect requests into a batch."""
        batch = []
        deadline = time.time() + (self.timeout_ms / 1000.0)
        
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0 and batch:
                break
            
            try:
                timeout = max(0.001, remaining) if batch else None
                _, _, request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_batch(self, batch: List[TTSRequest]) -> None:
        """Process a batch of requests."""
        start_time = time.time()
        
        try:
            results = await self.process_fn(batch)
            
            processing_time = (time.time() - start_time) * 1000
            self._total_batches += 1
            self._total_processing_time += processing_time
            
            logger.debug(
                f"Processed batch of {len(batch)} requests "
                f"in {processing_time:.1f}ms"
            )
            
            # Set results
            for request, result in zip(batch, results):
                if request.future and not request.future.done():
                    request.future.set_result(result)
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set errors on all futures
            for request in batch:
                if request.future and not request.future.done():
                    request.future.set_exception(e)
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        avg_batch_time = (
            self._total_processing_time / self._total_batches
            if self._total_batches > 0 else 0
        )
        avg_batch_size = (
            self._total_requests / self._total_batches
            if self._total_batches > 0 else 0
        )
        
        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "avg_batch_size": avg_batch_size,
            "avg_batch_time_ms": avg_batch_time,
            "queue_size": self._queue.qsize() if self._queue else 0
        }


class SimpleQueue:
    """
    Simple thread-safe queue for synchronous processing.
    
    Used when asyncio is not available or for simpler use cases.
    """
    
    def __init__(self, maxsize: int = 100):
        self._queue = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
    
    def put(self, item: Any) -> None:
        """Add item to queue."""
        with self._not_empty:
            self._queue.append(item)
            self._not_empty.notify()
    
    def get(self, timeout: Optional[float] = None) -> Any:
        """Get item from queue."""
        with self._not_empty:
            if not self._queue:
                self._not_empty.wait(timeout=timeout)
            if self._queue:
                return self._queue.popleft()
            return None
    
    def qsize(self) -> int:
        """Get queue size."""
        return len(self._queue)
