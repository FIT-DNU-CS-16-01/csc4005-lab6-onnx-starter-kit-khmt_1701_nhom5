import os
import json
import torch
import torch.nn as nn
import torchvision.models as models

# =========================
# Create output directory
# =========================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load ViT model
# =========================
model = models.vit_b_16(weights=None)

# IndoorCVPR có 67 class
num_classes = 67

model.heads.head = nn.Linear(
    model.heads.head.in_features,
    num_classes
)

model.eval()

# =========================
# Dummy input
# =========================
dummy_input = torch.randn(1, 3, 224, 224)

# =========================
# Export ONNX
# =========================
onnx_path = os.path.join(
    OUTPUT_DIR,
    "vit_indoorcvpr.onnx"
)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print(f"ONNX model saved to: {onnx_path}")

# =========================
# Save report
# =========================
report = {
    "model": "ViT_B_16",
    "num_classes": num_classes,
    "opset_version": 17,
    "dynamic_batch": True
}

report_path = os.path.join(
    OUTPUT_DIR,
    "export_report.json"
)

with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print(f"Report saved to: {report_path}")