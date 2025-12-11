"""
API Integration Tests

Tests for the TTS server endpoints.
"""

import pytest
import io
from fastapi.testclient import TestClient


# Skip tests if dependencies not available
pytest.importorskip("fastapi")


class TestHealthEndpoints:
    """Tests for health and metrics endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "speakers_cached" in data


class TestSpeakerEndpoints:
    """Tests for speaker management endpoints."""
    
    def test_list_speakers(self, client):
        """Test list speakers endpoint."""
        response = client.get("/v1/speakers")
        assert response.status_code == 200
        
        data = response.json()
        assert "speakers" in data
        assert "total" in data
        assert isinstance(data["speakers"], list)


class TestTTSEndpoints:
    """Tests for TTS endpoints."""
    
    def test_synthesize_endpoint_exists(self, client):
        """Test that synthesize endpoint exists."""
        # This will fail with 503 if model not loaded, but endpoint should exist
        response = client.post("/v1/tts/synthesize", json={
            "text": "Hello world",
            "speed": 1.0
        })
        # Accept 200 (success) or 503 (model not loaded) or 500 (missing speaker)
        assert response.status_code in [200, 500, 503]
    
    def test_clone_endpoint_exists(self, client):
        """Test that clone endpoint exists."""
        # Create a minimal audio file
        audio_data = b'\x00' * 1000
        
        response = client.post("/v1/tts/clone", data={
            "text": "Hello world",
            "prompt_text": "Test prompt"
        }, files={
            "prompt_audio": ("test.wav", io.BytesIO(audio_data), "audio/wav")
        })
        # Accept various status codes, just checking endpoint exists
        assert response.status_code in [200, 400, 500, 503]
    
    def test_stream_endpoint_exists(self, client):
        """Test that stream endpoint exists."""
        response = client.post("/v1/tts/stream", data={
            "text": "Hello world"
        })
        assert response.status_code in [200, 500, 503]


class TestLegacyEndpoints:
    """Tests for legacy CosyVoice-compatible endpoints."""
    
    def test_legacy_zero_shot(self, client):
        """Test legacy zero_shot endpoint exists."""
        audio_data = b'\x00' * 1000
        
        response = client.post("/inference_zero_shot", data={
            "tts_text": "Hello",
            "prompt_text": "Test"
        }, files={
            "prompt_wav": ("test.wav", io.BytesIO(audio_data), "audio/wav")
        })
        assert response.status_code in [200, 400, 500, 503]


@pytest.fixture
def client():
    """Create test client."""
    import sys
    from pathlib import Path
    
    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Mock the engine to avoid loading the model
    import os
    os.environ["LOAD_MODEL_ON_STARTUP"] = "false"
    
    from server.main import app
    return TestClient(app)
