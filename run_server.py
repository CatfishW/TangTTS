#!/usr/bin/env python3
"""
CosyVoice2 TTS Server Launcher

CLI entry point with configuration options.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import yaml
import uvicorn


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_environment(config: dict):
    """Set up environment variables from config."""
    model_config = config.get('model', {})
    server_config = config.get('server', {})
    cache_config = config.get('cache', {})
    
    os.environ.setdefault('MODEL_PATH', model_config.get('path', 'pretrained_models/CosyVoice2-0.5B'))
    os.environ.setdefault('USE_VLLM', str(model_config.get('use_vllm', True)).lower())
    os.environ.setdefault('USE_JIT', str(model_config.get('use_jit', True)).lower())
    os.environ.setdefault('USE_TRT', str(model_config.get('use_trt', False)).lower())
    os.environ.setdefault('USE_FP16', str(model_config.get('fp16', True)).lower())
    os.environ.setdefault('SPEAKER_CACHE_SIZE', str(cache_config.get('speaker_cache_size', 100)))
    os.environ.setdefault('SPEAKER_PERSIST_PATH', cache_config.get('speaker_persist_path', 'data/speakers'))


def main():
    parser = argparse.ArgumentParser(
        description='CosyVoice2 High-Performance TTS Server'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Server host (overrides config)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=None,
        help='Server port (overrides config)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of workers (overrides config)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model directory'
    )
    parser.add_argument(
        '--no-vllm',
        action='store_true',
        help='Disable vLLM acceleration'
    )
    parser.add_argument(
        '--no-jit',
        action='store_true',
        help='Disable JIT compilation'
    )
    parser.add_argument(
        '--trt',
        action='store_true',
        help='Enable TensorRT acceleration'
    )
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Disable FP16 inference'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error'],
        help='Logging level'
    )
    parser.add_argument(
        '--download-model',
        action='store_true',
        help='Download model before starting server'
    )
    
    args = parser.parse_args()
    
    # Load config file if exists
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
        setup_environment(config)
    
    # Override with CLI arguments
    if args.model_path:
        os.environ['MODEL_PATH'] = args.model_path
    if args.no_vllm:
        os.environ['USE_VLLM'] = 'false'
    if args.no_jit:
        os.environ['USE_JIT'] = 'false'
    if args.trt:
        os.environ['USE_TRT'] = 'true'
    if args.no_fp16:
        os.environ['USE_FP16'] = 'false'
    
    # Download model if requested
    if args.download_model:
        print("Downloading CosyVoice2-0.5B model...")
        from modelscope import snapshot_download
        model_path = os.environ.get('MODEL_PATH', 'pretrained_models/CosyVoice2-0.5B')
        snapshot_download('iic/CosyVoice2-0.5B', local_dir=model_path)
        print(f"Model downloaded to {model_path}")
    
    # Get server settings
    server_config = config.get('server', {})
    host = args.host or server_config.get('host', '0.0.0.0')
    port = args.port or server_config.get('port', 8000)
    workers = args.workers or server_config.get('workers', 1)
    
    # Configure logging
    log_config = config.get('logging', {})
    log_level = args.log_level.upper() or log_config.get('level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          CosyVoice2 High-Performance TTS Server              ║
╠══════════════════════════════════════════════════════════════╣
║  Model: {os.environ.get('MODEL_PATH', 'pretrained_models/CosyVoice2-0.5B'):<50} ║
║  vLLM:  {os.environ.get('USE_VLLM', 'true'):<50} ║
║  JIT:   {os.environ.get('USE_JIT', 'true'):<50} ║
║  TRT:   {os.environ.get('USE_TRT', 'false'):<50} ║
║  FP16:  {os.environ.get('USE_FP16', 'true'):<50} ║
╠══════════════════════════════════════════════════════════════╣
║  Server: http://{host}:{port:<43} ║
║  API Docs: http://{host}:{port}/docs{' ':<36} ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Start server
    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == '__main__':
    main()
