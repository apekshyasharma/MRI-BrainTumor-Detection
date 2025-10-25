# MRI Brain Tumor Classification (CNN + Transfer Learning)

A reproducible workflow for classifying brain MRI scans into four categories:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

This repository centers around a single notebook: [MRIBrainTumorDetection.ipynb](MRIBrainTumorDetection.ipynb), which walks through:
- S1: Image preprocessing and loading with Keras `ImageDataGenerator`
- S2: A baseline CNN built from scratch
- S3: Transfer learning with VGG16
- S4: Fine-tuning, evaluation, and visualization

> Medical disclaimer: This project is for research and educational purposes only and is not intended for clinical use.

---

## Features

- End-to-end pipeline in one notebook
- Augmented training with Keras generators
- Transfer learning using VGG16 with optional fine-tuning
- Early stopping and training curves
- Confusion matrix, classification report, and sample predictions

---

## Dataset

- Source: Brain Tumor MRI Dataset (Kaggle)
- URL: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

Expected directory structure:
```
datasets/
  Training/
    glioma_tumor/
    meningioma_tumor/
    pituitary_tumor/
    no_tumor/
  Testing/
    glioma_tumor/
    meningioma_tumor/
    pituitary_tumor/
    no_tumor/
```

In the notebook, set:
- train_path = '/content/drive/MyDrive/datasets/Training'
- test_path  = '/content/drive/MyDrive/datasets/Testing'

Update paths if running locally.

---

## Environment

- Python 3.9+
- TensorFlow 2.10+ (GPU recommended)
- NumPy, Matplotlib, scikit-learn
- Jupyter or VS Code with Python extension

Install dependencies:
```sh
python -m venv .venv
. .venv/Scripts/activate    # Windows
# source .venv/bin/activate # macOS/Linux
pip install --upgrade pip
pip install tensorflow numpy matplotlib scikit-learn jupyter
```

---

## Quick Start

### Option A: Google Colab (recommended)
1. Upload this repo or open the notebook in Colab.
2. Mount Google Drive in the notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```
3. Place the dataset under `/content/drive/MyDrive/datasets/`.
4. Run all cells in [MRIBrainTumorDetection.ipynb](MRIBrainTumorDetection.ipynb).

### Option B: Local (VS Code/Jupyter)
1. Place the dataset under `datasets/Training` and `datasets/Testing`.
2. Open [MRIBrainTumorDetection.ipynb](MRIBrainTumorDetection.ipynb) in VS Code or Jupyter.
3. Update `train_path` and `test_path` to your local folders.
4. Run the notebook cells in order.

---

## Training

The notebook contains two training flows:

- Baseline CNN:
  - Small 3-block CNN
  - Categorical cross-entropy, Adam optimizer
  - 224×224 image size

- Transfer Learning (VGG16):
  - ImageNet weights, `include_top=False`
  - GlobalAveragePooling + Dense classifier head
  - Phase 1: freeze base, train head
  - Phase 2: unfreeze top layers, fine-tune with lower LR
  - Early stopping on validation loss

Adjust `epochs`, `batch_size`, and augmentation params as needed.

---

## Evaluation

The notebook provides:
- Test accuracy via `model.evaluate()`
- Confusion matrix and classification report (precision, recall, F1)
- Visualization of predictions vs. ground truth

Example metric definition:
- Accuracy: $Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$

---

## Inference and Saving

Add this at the end of the notebook to save your best model:
```python
model.save("brain_tumor_vgg16.h5")
```

To run inference on a single image, add a helper cell:
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

class_names = list(test_set.class_indices.keys())

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # match preprocessing used in training
    probs = model.predict(x)[0]
    pred_idx = np.argmax(probs)
    return class_names[pred_idx], float(probs[pred_idx])

label, confidence = predict_image("path/to/image.jpg")
print(f"Predicted: {label} ({confidence:.3f})")
```

---

## Repository Structure

```
.
├─ MRIBrainTumorDetection.ipynb   # Main end-to-end notebook
└─ README.md                      # This file
```

---

## Roadmap

- Add Grad-CAM visualizations for interpretability
- Export SavedModel/TF Lite for deployment
- Hyperparameter search (KerasTuner/Optuna)
- Add ResNet50 and EfficientNet baselines

---

## Acknowledgments

- Dataset by Sartaj Bhuvaji on Kaggle
- VGG16 from Keras Applications (ImageNet weights)

---

## License

This project is released under the MIT License. See LICENSE for details.

---

## Responsible AI/Medical Disclaimer

- Do not use this model for diagnosis or clinical decision-making.
- Performance can vary across institutions and imaging protocols.
- Always consult certified medical professionals for clinical use.
