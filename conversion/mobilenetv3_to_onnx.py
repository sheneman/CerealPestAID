# mobilenetv3_to_onnx.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
import onnx
from onnxsim import simplify

CKPT = "models/best_factai_mobilenetv3.pth"
OUT_ONNX = "models/factai_mobilenetv3.onnx"
SIMPLIFIED = "models/mobilenetv3_simplified.onnx"

NUM_CLASSES = 26
INPUT_SIZE = (224, 224)   # change if you trained at a different size

def load_state_dict_safely(path):
    sd = torch.load(path, map_location="cpu")
    # unwrap common training wrappers
    if isinstance(sd, dict):
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
    # strip 'module.' from DDP/DataParallel
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    # some trainers save as 'model.xxx' — strip that too
    sd = { (k[6:] if k.startswith("model.") else k): v for k, v in sd.items() }
    return sd

def main():
    model = mobilenet_v3_large(weights=None)
    # replace final classifier to match your class count
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)

    sd = load_state_dict_safely(CKPT)
    # use strict=False first to see what (if anything) doesn't match
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded checkpoint.\n Missing keys:", missing, "\n Unexpected keys:", unexpected)

    # if the only missing/unexpected were benign (e.g., none), you can tighten to strict=True later
    model.eval()

    dummy = torch.randn(1, 3, *INPUT_SIZE)
    torch.onnx.export(
        model, dummy, OUT_ONNX,
        input_names=["input"], output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    print("Exported ONNX ->", OUT_ONNX)

    m = onnx.load(OUT_ONNX)
    m_s, _ = simplify(m)
    onnx.save(m_s, SIMPLIFIED)
    print("Saved simplified ONNX ->", SIMPLIFIED)

if __name__ == "__main__":
    main()

