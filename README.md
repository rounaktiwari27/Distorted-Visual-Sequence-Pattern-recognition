# Distorted Sequence Pattern Recognition 

An end-to-end Deep Learning pipeline built to predict 6-character alphanumeric sequences from images subjected to severe synthetic distortion, occlusions, and high-frequency noise.

## Performance
* **Validation Character Error Rate (CER):** `0.0042` (99.58% Accuracy)
* **Architecture:** Custom Convolutional Recurrent Neural Network (CRNN)
* **Loss Function:** Connectionist Temporal Classification (CTC)

## 🧠 Engineering Constraints & Architecture
This project was built under constraints: **No pre-trained weights, no fine-tuning, and no external deep architectures (e.g., ResNet, ViT).** The model was engineered using foundational PyTorch primitives.

1. **CNN Feature Extractor:** A 5-layer convolutional stack with asymmetrical max-pooling. It aggressively downsamples the vertical dimension while preserving the horizontal time-steps required for sequence modeling.
2. **BiLSTM Sequence Modeler:** A 2-layer Bidirectional LSTM reads the spatial feature slices forward and backward, mathematically deducing characters hidden beneath severe black-ellipse occlusions based on contextual edges.
3. **CTC Loss & Greedy Decoder:** Aligns the predicted sequences with variable labels, completely bypassing the need for expensive character-level bounding box coordinates.

## Repository Structure
* `/src/`: Production-ready modular Python scripts (`config.py`, `dataset.py`, `model.py`, `train.py`, `infer.py`).
* `notebook_Rounak_24115125.ipynb`: The complete Data Story, proving model convergence, evaluation metrics, and deterministic inference.
* `/outputs/checkpoints/`: Contains the `best_model.pth` weights.

## How to Run
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt