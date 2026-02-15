# Model Conversion Guide

This directory contains scripts for converting trained PyTorch models to ONNX format (and subsequently to TFLite).

## PyTorch to ONNX

Each script loads a trained `.pth` checkpoint, reconstructs the model architecture, and exports to ONNX with graph simplification via `onnxsim`.

### EfficientNet-B6

```bash
python conversion/efficientnet_to_onnx.py
```

- Input: `models/best_factai_efficientnet-b6.pth`
- Output: `models/efficientnet-b6_simplified.onnx`
- Input size: 528x528
- Opset version: 13

### InceptionV3

```bash
python conversion/inceptionv3_to_onnx.py
```

- Input: `models/best_factai_inceptionv3.pth`
- Output: `models/inceptionv3_simplified.onnx`
- Input size: 299x299
- Opset version: 13

### MobileNetV3-Large

```bash
python conversion/mobilenetv3_to_onnx.py
```

- Input: `models/best_factai_mobilenetv3.pth`
- Output: `models/mobilenetv3_simplified.onnx`
- Input size: 224x224
- Opset version: 13

## ONNX to TFLite

TFLite conversion was performed using the `onnx-tf` library followed by the TensorFlow Lite converter:

```python
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# ONNX -> TensorFlow SavedModel
onnx_model = onnx.load("model_simplified.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")

# TensorFlow SavedModel -> TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("model_tf")
tflite_model = converter.convert()
with open("model_fp32.tflite", "wb") as f:
    f.write(tflite_model)
```

## Dependencies

```bash
pip install torch torchvision efficientnet_pytorch onnx onnxsim onnx-tf tensorflow
```
