import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

class Config:
    # ==========================================
    # 1. System & Paths Configuration
    # ==========================================
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SCRIPT_ROOT = Path(__file__).resolve().parent

    # Dataset Paths
    UK_AUDIO_DIR = Path(os.getenv("UK_AUDIO_DIR", str(PROJECT_ROOT / "data" / "UK_audio_preprocessed"))).resolve()
    ICBHI_AUDIO_DIR = Path(os.getenv("ICBHI_AUDIO_DIR", str(PROJECT_ROOT / "data" / "ICBHI_final_database"))).resolve()
    DEFAULT_TEST_AUDIO = Path(os.getenv("DEFAULT_TEST_AUDIO", PROJECT_ROOT / "data" / "test_audio" / "WheezingCough.wav")).resolve()

    # Output Paths
    CACHE_DIR = Path(os.getenv("CACHE_DIR", PROJECT_ROOT / "distillation_cache")).resolve()
    CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", SCRIPT_ROOT / "checkpoints")).resolve()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 2. Hardware Configuration
    # ==========================================
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    # ==========================================
    # 3. Audio DSP Parameters
    # ==========================================
    SAMPLE_RATE = 16000
    N_FFT = 1024
    HOP_LENGTH = 256
    N_MELS = 80
    F_MIN = 0.0
    F_MAX = 8000.0

    TARGET_SAMPLES = 40000 
    TARGET_FRAMES = 157    

    # ==========================================
    # 4. Model & Training Hyperparameters
    # ==========================================
    MODEL_NAME = os.getenv("MODEL_NAME", "mobilenetv3_small_100")
    IN_CHANNELS = 3
    NUM_CLASSES = 3  # [0: Wheezing, 1: Duration, 2: SoundType]

    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
    EPOCHS = int(os.getenv("EPOCHS", "30"))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-3"))

    PRODUCER_NAME = "CCBaby_Export_Pipeline"
    ONNX_OPSET = 14