# CerealPestAID

Training and evaluation of deep learning models for cereal pest identification (26 species).

## Overview

CerealPestAID provides trained deep learning classifiers for identifying 26 species of cereal crop pests from images. Three model architectures are compared: **EfficientNet-B6**, **MobileNetV3-Large**, and **InceptionV3**. Pre-trained model weights are available in PyTorch, ONNX, and TFLite formats.

## Dataset

The dataset contains labeled images of 26 cereal pest species organized into train/val/test splits using `torchvision.datasets.ImageFolder` conventions (one subdirectory per class).

| Split | Samples |
|-------|---------|
| Train | ~80%    |
| Val   | ~10%    |
| Test  | 2,381   |

**26 Classes:**

| Index | Species |
|-------|---------|
| 0 | Cabbage seedpod weevil |
| 1 | Bird cherry oat aphid |
| 2 | Cabbage aphid |
| 3 | Cereal grass aphid |
| 4 | Cereal leaf beetle |
| 5 | Clickbeetles / wireworms |
| 6 | Crucifer flea beetle |
| 7 | Cutworms |
| 8 | Diamondback moth |
| 9 | English grain aphid |
| 10 | Greenbug |
| 11 | Green peach aphid |
| 12 | Hessian fly |
| 13 | Lygus bug |
| 14 | Non-pest herbivores |
| 15 | Occasional pest |
| 16 | Pea aphid |
| 17 | Pea leaf weevil |
| 18 | Pea weevil |
| 19 | Predators |
| 20 | Rose grain aphid |
| 21 | Russian wheat aphid |
| 22 | Stink bug |
| 23 | Striped flea beetle |
| 24 | Turnip aphid |
| 25 | Wheathead armyworm (*Dargida diffusa*) |

The dataset is available on HuggingFace: [sheneman/CerealPestAID-dataset](https://huggingface.co/datasets/sheneman/CerealPestAID-dataset)

## Model Performance

Evaluated on the held-out test set (2,381 samples):

| Model | Test Accuracy | Parameters | Best Classes (100%) | Weakest Class |
|-------|:------------:|:----------:|:-------------------:|---------------|
| **EfficientNet-B6** | **92.94%** | ~43M | 7 classes | non_pest_herbivores (66.7%) |
| MobileNetV3-Large | 90.09% | ~5.4M | 3 classes | non_pest_herbivores (60.0%) |
| InceptionV3 | 79.00% | ~27M | 2 classes | cabbage_aphid (46.4%) |

### Confusion Matrices

| EfficientNet-B6 | MobileNetV3-Large | InceptionV3 |
|:---:|:---:|:---:|
| ![EfficientNet-B6](results/confusion_matrix_efficientnet.png) | ![MobileNetV3](results/confusion_matrix_mobilenetv3.png) | ![InceptionV3](results/confusion_matrix_inceptionv3.png) |

## Training Details

All models were trained with the following configuration:

- **Optimizer:** Adam (lr=1e-4)
- **LR Schedule:** ReduceLROnPlateau (factor=0.9, patience=5)
- **Batch size:** 8
- **Max epochs:** 1000 (early stopping via best validation loss)
- **Loss:** CrossEntropyLoss
- **Class balancing:** WeightedRandomSampler (inverse class frequency)
- **Input resolution:** 528x528 (resize to 572, then crop)
- **Hardware:** NVIDIA GPU via SLURM (CUDA 11.8)

### Data Augmentation (training)
- RandomResizedCrop (528, scale 0.6-1.0)
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation (30 degrees)
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Repository Structure

```
CerealPestAID/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── label_mappings.txt
├── training/
│   ├── train_efficientnet.py
│   ├── train_inceptionv3.py
│   └── train_mobilenetv3.py
├── evaluation/
│   ├── eval_efficientnet.py
│   ├── eval_inceptionv3.py
│   └── eval_mobilenetv3.py
├── conversion/
│   ├── efficientnet_to_onnx.py
│   ├── inceptionv3_to_onnx.py
│   ├── mobilenetv3_to_onnx.py
│   └── CONVERSION_GUIDE.md
├── inference/
│   ├── infer_tflite.py
│   └── heatmap.py
├── slurm/
│   ├── train_efficientnet.slurm
│   ├── train_inceptionv3.slurm
│   └── train_mobilenetv3.slurm
└── results/
    ├── confusion_matrix_efficientnet.png
    ├── confusion_matrix_inceptionv3.png
    └── confusion_matrix_mobilenetv3.png
```

## Getting Started

### Installation

```bash
git clone https://github.com/sheneman/CerealPestAID.git
cd CerealPestAID
pip install -r requirements.txt
```

### Download Models

Pre-trained weights are hosted on HuggingFace: [sheneman/CerealPestAID](https://huggingface.co/sheneman/CerealPestAID)

```bash
pip install huggingface-hub
huggingface-cli download sheneman/CerealPestAID --local-dir ./hf_models
```

### Training

Prepare your dataset in ImageFolder format under `./data/labels/{train,val,test}/` with one subdirectory per class, then:

```bash
python training/train_efficientnet.py
python training/train_inceptionv3.py
python training/train_mobilenetv3.py
```

Or submit via SLURM:

```bash
sbatch slurm/train_efficientnet.slurm
```

### Evaluation

```bash
python evaluation/eval_efficientnet.py --test-dir ./data/labels/test --checkpoint ./models/best_factai_efficientnet-b6.pth
python evaluation/eval_inceptionv3.py --test-dir ./data/labels/test --checkpoint ./models/best_factai_inceptionv3.pth
python evaluation/eval_mobilenetv3.py --test-dir ./data/labels/test --checkpoint ./models/best_factai_mobilenetv3.pth
```

### Inference (TFLite)

```bash
python inference/infer_tflite.py path/to/image.jpg
```

### Model Conversion

See [conversion/CONVERSION_GUIDE.md](conversion/CONVERSION_GUIDE.md) for details on converting PyTorch checkpoints to ONNX and TFLite.

## Model Formats

Each model is available in three formats:

| Format | Use Case | Files |
|--------|----------|-------|
| PyTorch (.pth) | Training, fine-tuning, GPU inference | `best_factai_*.pth` |
| ONNX (.onnx) | Cross-platform inference, optimization | `*_simplified.onnx` |
| TFLite (.tflite) | Mobile/edge deployment | `*_fp32.tflite` |

## Links

- **Models (HuggingFace):** [sheneman/CerealPestAID](https://huggingface.co/sheneman/CerealPestAID)
- **Dataset (HuggingFace):** [sheneman/CerealPestAID-dataset](https://huggingface.co/datasets/sheneman/CerealPestAID-dataset)

## Author

Luke Sheneman
Institute for Interdisciplinary Data Sciences (IIDS)
University of Idaho
sheneman@uidaho.edu

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
