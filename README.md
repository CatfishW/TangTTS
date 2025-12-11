# CosyVoice2 High-Performance TTS Server

A highly optimized Text-to-Speech server based on CosyVoice2-0.5B with vLLM acceleration.

## Features

- 🚀 **vLLM Acceleration**: 2-3x faster LLM inference with PagedAttention
- ⚡ **TensorRT Support**: 4x faster flow decoder
- 🎤 **Zero-Shot Voice Cloning**: Clone any voice with 3-5 second prompt
- 🌊 **Streaming Support**: Real-time audio streaming via WebSocket
- 🌍 **Multilingual**: Chinese, English, Japanese, Korean, dialects
- 📊 **Production Ready**: Prometheus metrics, health checks

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download model (Hugging Face - recommended)
python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B', local_dir_use_symlinks=False)"

# Or use ModelScope (alternative)
# python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

# Start server
python run_server.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/tts/synthesize` | Text-to-speech synthesis |
| POST | `/v1/tts/clone` | Zero-shot voice cloning |
| WS | `/v1/tts/ws` | WebSocket streaming |
| POST | `/v1/speakers/register` | Register speaker |
| GET | `/v1/speakers` | List speakers |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

## Docker Deployment

```bash
docker-compose up -d
```

## License

Apache-2.0 (following CosyVoice license)
