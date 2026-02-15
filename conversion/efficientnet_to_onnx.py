# efficientnet_pytorch_b6_to_onnx.py
import torch
import onnx
from onnxsim import simplify
from efficientnet_pytorch import EfficientNet

CKPT = "models/best_factai_efficientnet-b6.pth"
OUT_ONNX = "models/best_factai_efficientnet-b6.onnx"
SIMPLIFIED_ONNX = "models/efficientnet-b6_simplified.onnx"

ARCH = "efficientnet-b6"   # <- match your training arch exactly
NUM_CLASSES = 26           # <- match your checkpoint's class count
INPUT_SIZE = 528           # B6 default (use 528 unless you trained at a different resolution)

def load_state_dict_safely(path):
    sd = torch.load(path, map_location="cpu")
    # unwrap common wrappers
    if isinstance(sd, dict):
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
    # strip 'module.' from DDP/DataParallel
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    return sd

def main():
    # Build the same model you trained
    model = EfficientNet.from_name(ARCH, num_classes=NUM_CLASSES, in_channels=3)

    # IMPORTANT: disable memory-efficient swish for ONNX export
    if hasattr(model, "set_swish"):
        model.set_swish(memory_efficient=False)

    # Load weights
    sd = load_state_dict_safely(CKPT)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded checkpoint.\n Missing keys:", missing, "\n Unexpected keys:", unexpected)

    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # Export
    torch.onnx.export(
        model, dummy, OUT_ONNX,
        input_names=["input"], output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        training=torch.onnx.TrainingMode.EVAL,
    )
    print("Exported ONNX ->", OUT_ONNX)

    # Simplify graph
    m = onnx.load(OUT_ONNX)
    m_s, _ = simplify(m)
    onnx.save(m_s, SIMPLIFIED_ONNX)
    print("Saved simplified ONNX ->", SIMPLIFIED_ONNX)

if __name__ == "__main__":
    main()

