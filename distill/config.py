import os

# --- Data ---
BATCH_SIZE = 64
IMAGE_SIZE = 224
NUM_WORKERS = 8

# --- Optimization ---
LR = 1e-4
WEIGHT_DECAY = 0.05
MAX_STEPS = 100_000
WARMUP_STEPS = 2_000
GRAD_CLIP = 1.0

# --- Loss weights (AM-RADIO) ---
W_SUMMARY = 1.0
W_FEATURE = 1.0

# --- Validation / logging / checkpoints ---
VAL_CHECK_INTERVAL = 1_000
VAL_BATCHES = 20
LOG_EVERY = 50
CKPT_DIR = "checkpoints"

# Precision: "auto" -> bf16 on GPU, fp32 on CPU. Override with a Lightning string.
PRECISION = "auto"


def resolve_precision():
    if PRECISION != "auto":
        return PRECISION
    import torch

    return "bf16-mixed" if torch.cuda.is_available() else "32-true"


def maybe_login():
    """huggingface_hub login from env. Never hardcode the token."""
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token)
    else:
        print("WARNING: HF_TOKEN not set; gated teacher downloads will fail.")
