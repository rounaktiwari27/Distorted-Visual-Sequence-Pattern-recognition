# Distorted Visual Sequence Pattern Recognition

A custom Deep Learning pipeline for recognizing 6-character alphanumeric sequences from heavily distorted grayscale images. The project tackles a challenging OCR problem involving severe occlusions, background noise, blur artifacts, overlapping characters, and irregular spacing.

The objective is to accurately reconstruct the hidden text sequence while minimizing Character Error Rate (CER), the official evaluation metric for the challenge.

---

## Project Overview

- No pre-trained models
- No transfer learning
- No external OCR frameworks
- Custom architecture built using PyTorch

To solve the problem, a Convolutional Recurrent Neural Network (CRNN) was designed and trained from scratch. The architecture combines convolutional feature extraction, bidirectional sequence modeling, and Connectionist Temporal Classification (CTC) for sequence prediction.

The final model is capable of recovering text even when large portions of characters are hidden beneath synthetic distortions and occlusions.

---

## Model Architecture

### CNN Feature Extractor

A custom 5-layer convolutional backbone extracts visual features from grayscale images.

Key design choices:

- Convolution + BatchNorm + ReLU blocks
- Progressive spatial downsampling
- Asymmetrical max-pooling to preserve horizontal sequence information

The CNN compresses the image height while retaining the width-based temporal features required for sequence recognition.

---

### BiLSTM Sequence Modeler

The extracted feature maps are converted into a sequence representation and passed through a 2-layer Bidirectional LSTM.

Advantages:

- Captures both left-to-right and right-to-left context
- Helps infer partially occluded characters
- Improves robustness against severe distortions

---

### CTC Loss and Decoding

Connectionist Temporal Classification (CTC) enables sequence prediction without requiring character-level segmentation or bounding box annotations.

Inference is performed using greedy decoding:

1. Select the most probable class at each timestep
2. Collapse consecutive duplicate predictions
3. Remove blank tokens
4. Recover the final text sequence

---

## Performance

| Metric | Value |
|----------|----------|
| Validation CER | **0.0042** |
| Character Accuracy | **99.58%** |
| Architecture | Custom CRNN |
| Loss Function | CTC Loss |

---

## Dataset Challenges

The dataset contains synthetically distorted text images affected by:

- Large black occlusion blobs
- Background noise
- Blur artifacts
- Character overlap
- Shape deformation
- Irregular spacing and alignment

These distortions make traditional OCR approaches unreliable and require contextual sequence modeling.

---

## Repository Structure

```text
Distorted-Text-Recognition/

├── data/
│   ├── train/
│   │   ├── train_images/
│   │   └── train-labels.csv
│   │
│   └── test/
│
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pth
│   │
│   ├── logs/
│   │   └── training_log.csv
│   │
│   └── predictions/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── utils.py
│   ├── model.py
│   ├── train.py
│   └── infer.py
│
├── notebook_Rounak_Tiwari_24115125.ipynb
├── submission_Rounak_Tiwari_24115125.csv
├── test_setup.py
├── requirements.txt
├── README.md
└── .gitignore