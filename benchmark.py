#!/usr/bin/env python3
"""
TTS Server Benchmark Script

Measures throughput, latency, and RTF (Real-Time Factor).
"""

import os
import sys
import time
import argparse
import asyncio
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import List
from concurrent.futures import ThreadPoolExecutor

import httpx


@dataclass
class BenchmarkResult:
    """Result of a single benchmark request."""
    request_id: int
    text_length: int
    audio_duration_seconds: float
    response_time_ms: float
    success: bool
    error: str = ""


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    total_audio_seconds: float
    total_processing_seconds: float
    rtf: float  # Real-Time Factor
    throughput_rps: float  # Requests per second


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


async def make_request(
    client: httpx.AsyncClient,
    url: str,
    text: str,
    request_id: int
) -> BenchmarkResult:
    """Make a single TTS request."""
    start_time = time.time()
    
    try:
        response = await client.post(
            url,
            json={"text": text, "speed": 1.0},
            timeout=60.0
        )
        response_time_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            # Estimate audio duration (WAV header + samples at 22050Hz)
            audio_size = len(response.content)
            audio_samples = (audio_size - 44) / 2  # 16-bit samples
            audio_duration = audio_samples / 22050
            
            return BenchmarkResult(
                request_id=request_id,
                text_length=len(text),
                audio_duration_seconds=audio_duration,
                response_time_ms=response_time_ms,
                success=True
            )
        else:
            return BenchmarkResult(
                request_id=request_id,
                text_length=len(text),
                audio_duration_seconds=0,
                response_time_ms=response_time_ms,
                success=False,
                error=f"HTTP {response.status_code}"
            )
            
    except Exception as e:
        return BenchmarkResult(
            request_id=request_id,
            text_length=len(text),
            audio_duration_seconds=0,
            response_time_ms=(time.time() - start_time) * 1000,
            success=False,
            error=str(e)
        )


async def run_benchmark(
    base_url: str,
    texts: List[str],
    concurrent: int,
    warmup: int = 3
) -> BenchmarkSummary:
    """Run the benchmark."""
    url = f"{base_url}/v1/tts/synthesize"
    results: List[BenchmarkResult] = []
    
    async with httpx.AsyncClient() as client:
        # Warmup requests
        if warmup > 0:
            print(f"Running {warmup} warmup requests...")
            for i in range(warmup):
                await make_request(client, url, texts[i % len(texts)], -1)
            print("Warmup complete")
        
        # Benchmark requests
        print(f"\nRunning {len(texts)} benchmark requests (concurrent={concurrent})...")
        start_time = time.time()
        
        semaphore = asyncio.Semaphore(concurrent)
        
        async def limited_request(text: str, request_id: int):
            async with semaphore:
                return await make_request(client, url, text, request_id)
        
        tasks = [
            limited_request(text, i) 
            for i, text in enumerate(texts)
        ]
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            
            # Progress update
            if (i + 1) % 10 == 0 or i == len(tasks) - 1:
                print(f"  Progress: {i + 1}/{len(tasks)}")
        
        total_time = time.time() - start_time
    
    # Calculate summary
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    response_times = [r.response_time_ms for r in successful]
    
    if not response_times:
        response_times = [0.0]
    
    total_audio = sum(r.audio_duration_seconds for r in successful)
    
    return BenchmarkSummary(
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        avg_response_time_ms=statistics.mean(response_times),
        p50_response_time_ms=percentile(response_times, 50),
        p95_response_time_ms=percentile(response_times, 95),
        p99_response_time_ms=percentile(response_times, 99),
        min_response_time_ms=min(response_times),
        max_response_time_ms=max(response_times),
        total_audio_seconds=total_audio,
        total_processing_seconds=total_time,
        rtf=total_time / total_audio if total_audio > 0 else 0,
        throughput_rps=len(successful) / total_time if total_time > 0 else 0
    )


def print_summary(summary: BenchmarkSummary):
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"""
Requests:
  Total:      {summary.total_requests}
  Successful: {summary.successful_requests}
  Failed:     {summary.failed_requests}

Response Time (ms):
  Average:    {summary.avg_response_time_ms:.1f}
  P50:        {summary.p50_response_time_ms:.1f}
  P95:        {summary.p95_response_time_ms:.1f}
  P99:        {summary.p99_response_time_ms:.1f}
  Min:        {summary.min_response_time_ms:.1f}
  Max:        {summary.max_response_time_ms:.1f}

Performance:
  Total audio:      {summary.total_audio_seconds:.1f} seconds
  Total time:       {summary.total_processing_seconds:.1f} seconds
  RTF:              {summary.rtf:.3f} (lower is better, <1 = faster than real-time)
  Throughput:       {summary.throughput_rps:.2f} requests/second
""")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TTS Server")
    parser.add_argument(
        "--url", "-u",
        type=str,
        default="http://localhost:8000",
        help="Base URL of TTS server"
    )
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=100,
        help="Number of requests to make"
    )
    parser.add_argument(
        "--concurrent", "-c",
        type=int,
        default=10,
        help="Number of concurrent requests"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=3,
        help="Number of warmup requests"
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="File containing test texts (one per line)"
    )
    args = parser.parse_args()
    
    # Default test texts
    default_texts = [
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Speech synthesis is the artificial production of human speech.",
        "CosyVoice is a multilingual voice generation model.",
        "This benchmark measures throughput and latency performance.",
        "Natural language processing enables computers to understand human language.",
        "Machine learning models can generate realistic human speech.",
        "Real-time audio streaming requires low latency processing.",
        "Voice cloning technology can replicate unique vocal characteristics.",
        "The future of AI includes seamless human-computer interaction.",
    ]
    
    # Load texts from file if provided
    if args.text_file:
        with open(args.text_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = default_texts
    
    # Expand to requested number
    full_texts = []
    for i in range(args.requests):
        full_texts.append(texts[i % len(texts)])
    
    print(f"CosyVoice TTS Benchmark")
    print(f"Server: {args.url}")
    print(f"Requests: {args.requests}")
    print(f"Concurrent: {args.concurrent}")
    
    # Run benchmark
    summary = asyncio.run(run_benchmark(
        args.url,
        full_texts,
        args.concurrent,
        args.warmup
    ))
    
    print_summary(summary)


if __name__ == "__main__":
    main()
