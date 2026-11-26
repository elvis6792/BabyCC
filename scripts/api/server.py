import io
import asyncio
import soundfile as sf
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import Config
from scripts.inference_engine import AsthmaInferencePipeline

engine_instance = None
MAX_FILE_SIZE = 10 * 1024 * 1024

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_instance
    model_path = Config.CHECKPOINT_DIR / f"{Config.MODEL_NAME}_3d_production.onnx"
    
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}")
        
    engine_instance = AsthmaInferencePipeline(str(model_path))
    yield
    engine_instance = None

app = FastAPI(
    title="Asthma/COPD 3D Inference API",
    version="1.0.0",
    lifespan=lifespan
)

def process_audio_sync(file_bytes: bytes) -> dict:
    try:
        y, sr = sf.read(io.BytesIO(file_bytes))
        
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        if sr != Config.SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
            sr = Config.SAMPLE_RATE
            
        target_samples = Config.TARGET_SAMPLES
        current_samples = len(y)
        
        if current_samples < target_samples:
            # 長度不足：尾部補零 (靜音)
            y = np.pad(y, (0, target_samples - current_samples), mode='constant')
        elif current_samples > target_samples:
            # 長度超出：強制截斷
            y = y[:target_samples]
            
        y = y.astype(np.float32)
        result = engine_instance.predict(y, sr)
        return result
    except Exception as e:
        raise ValueError(f"Audio processing failed: {str(e)}")

@app.post("/api/v1/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds the 10MB limit.")
    
    try:
        result = await asyncio.to_thread(process_audio_sync, file_bytes)
        return JSONResponse(content=result)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error during inference.")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_loaded": engine_instance is not None}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1
    )