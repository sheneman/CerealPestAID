#!/usr/bin/env python3
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# ===============================================================
# Usage: python infer_tflite.py image.jpg
# ===============================================================

if len(sys.argv) < 2:
    print("Usage: python infer_tflite.py <image_path>")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]
MODEL_PATH = "models/efficientnet_b6_fp32.tflite"

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
input_shape = input_details[0]['shape']  # e.g., [1, 528, 528, 3] for EfficientNet-B6
height, width = input_shape[1], input_shape[2]

image = Image.open(IMAGE_PATH).convert('RGB').resize((width, height))
input_data = np.asarray(image, dtype=np.float32) / 255.0
input_data = np.expand_dims(input_data, axis=0)

# Set the tensor and run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get the output
output_data = interpreter.get_tensor(output_details[0]['index'])
pred = np.squeeze(output_data)

# Print top-5 predicted class indices and probabilities
top_k = pred.argsort()[-5:][::-1]
print("\nTop 5 predictions:")
for i in top_k:
    print(f"Class {i}: {pred[i]:.4f}")

