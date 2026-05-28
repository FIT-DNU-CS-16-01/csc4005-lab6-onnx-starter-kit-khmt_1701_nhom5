import os
import torch

from src.model import build_model

# =========================
# Create checkpoint folder
# =========================
os.makedirs("checkpoints", exist_ok=True)

# =========================
# Build exact repo model
# =========================
model = build_model(
    model_name="vit_b_16",
    num_classes=5,
    dropout=0.2,
    pretrained=False,
    train_mode="head_only",
)

# =========================
# Save checkpoint
# =========================
checkpoint_path = "checkpoints/best_model.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_idx": {
            "class0": 0,
            "class1": 1,
            "class2": 2,
            "class3": 3,
            "class4": 4,
        },
    },
    checkpoint_path,
)

print(f"Checkpoint saved to: {checkpoint_path}")