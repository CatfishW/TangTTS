# CosyVoice2 TTS Server with vLLM
# Multi-stage optimized Docker image

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    sox \
    libsox-dev \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install Miniconda for pynini
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN conda create -n cosyvoice python=3.10 -y \
    && conda install -n cosyvoice -c conda-forge pynini==2.1.5 -y \
    && conda clean -afy

# Activate conda environment by default
SHELL ["conda", "run", "-n", "cosyvoice", "/bin/bash", "-c"]

# ==================== Builder Stage ====================
FROM base AS builder

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir vllm==0.9.0 transformers==4.51.3 \
    && pip install --no-cache-dir -r requirements.txt

# Clone CosyVoice repository
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice \
    && cd /app/CosyVoice \
    && git submodule update --init --recursive

# ==================== Production Stage ====================
FROM base AS production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /app/CosyVoice /app/CosyVoice

# Copy application code
COPY . /app/tts-server/

# Set working directory
WORKDIR /app/tts-server

# Add CosyVoice to Python path
ENV PYTHONPATH=/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH

# Create directories
RUN mkdir -p /app/tts-server/pretrained_models \
    && mkdir -p /app/tts-server/data/speakers

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["conda", "run", "-n", "cosyvoice", "--no-capture-output", \
     "python", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]
