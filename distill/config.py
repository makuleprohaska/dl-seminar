import os


def _env_int(name, default):
    """Read an int knob from the environment, falling back to the default."""
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


# --- Data ---
BATCH_SIZE = _env_int("BATCH_SIZE", 64)
IMAGE_SIZE = 224
NUM_WORKERS = _env_int("NUM_WORKERS", 8)

# --- Optimization ---
LR = _env_float("LR", 1e-4)
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 0.05)
MAX_STEPS = _env_int("MAX_STEPS", 100_000)
WARMUP_STEPS = _env_int("WARMUP_STEPS", 2_000)
GRAD_CLIP = _env_float("GRAD_CLIP", 1.0)

# --- Loss weights (AM-RADIO) ---
W_SUMMARY = _env_float("W_SUMMARY", 1.0)
W_FEATURE = _env_float("W_FEATURE", 1.0)

# --- Validation / logging / checkpoints ---
VAL_CHECK_INTERVAL = _env_int("VAL_CHECK_INTERVAL", 1_000)
VAL_BATCHES = _env_int("VAL_BATCHES", 20)
LOG_EVERY = _env_int("LOG_EVERY", 50)
CKPT_DIR = os.environ.get("CKPT_DIR", "checkpoints")

# Precision: "auto" -> bf16 on GPU, fp32 on CPU. Override with a Lightning string
# via the PRECISION env var (e.g. "16-mixed" on Volta cards lacking native bf16).
PRECISION = os.environ.get("PRECISION", "auto")


def resolve_precision():
    if PRECISION != "auto":
        return PRECISION
    import torch

    if not torch.cuda.is_available():
        return "32-true"
    # Native bf16 only exists on Ampere+ (CC >= 8.0). On older cards (e.g. Volta
    # V100, CC 7.0) bf16 is emulated and ~5x slower, so prefer fp16 tensor cores.
    major = torch.cuda.get_device_capability()[0]
    return "bf16-mixed" if major >= 8 else "16-mixed"


def maybe_login():
    """huggingface_hub login from env. Never hardcode the token."""
    # In offline mode the teacher weights must already be cached; `login()` does a
    # network whoami() call that raises under HF_HUB_OFFLINE, so skip it.
    if os.environ.get("HF_HUB_OFFLINE") in ("1", "true", "True"):
        print("HF_HUB_OFFLINE set; skipping login (using cached teacher weights).")
        return
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token)
    else:
        print("WARNING: HF_TOKEN not set; gated teacher downloads will fail.")
