"""Project uses PyTorch only; disable unused HF backends before test imports."""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
