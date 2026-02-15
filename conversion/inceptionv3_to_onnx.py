# inception2onnx.py
import torch
import torch.nn as nn
from torchvision.models import inception_v3
import onnx
from onnxsim import simplify

CKPT = "models/best_factai_inceptionv3.pth"
OUT_ONNX = "models/best_factai_inceptionv3.onnx"
NUM_CLASSES = 26            # adjust if needed
INPUT_SIZE = (299, 299)     # inception-v3 default

def load_state_dict_safely(path):
    sd = torch.load(path, map_location="cpu")
    # unwrap common training wrappers
    if isinstance(sd, dict):
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
    # strip 'module.' if saved from DataParallel/DDP
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def main():
    # Build model with aux logits if that’s how it was trained.
    # In eval() mode, forward returns only primary logits.
    model = inception_v3(weights=None, aux_logits=True)
    # replace final head to match class count (do this BEFORE loading if your checkpoint expects it)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    sd = load_state_dict_safely(CKPT)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded checkpoint. Missing:", missing)
    print("Unexpected:", unexpected)
    # If you still see lots of missing keys, your checkpoint likely doesn't match this architecture.

    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])

    torch.onnx.export(
        model, dummy, OUT_ONNX,
        input_names=["input"], output_names=["logits"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("Exported:", OUT_ONNX)

    m = onnx.load(OUT_ONNX)
    m_s, _ = simplify(m)
    onnx.save(m_s, "models/inceptionv3_simplified.onnx")
    print("Saved simplified ONNX -> models/inceptionv3_simplified.onnx")

if __name__ == "__main__":
    main()

