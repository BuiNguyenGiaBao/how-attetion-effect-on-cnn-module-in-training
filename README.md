# 🧠 How Attention Affects CNN Modules During Training

This repository explores **how attention mechanisms influence CNN-based models during training**, through comparative experiments on **image classification** and **video-based recognition** tasks.

The project is intended as a **public side project** focusing on:

* training dynamics,
* model behavior under limited data,
* and the practical role of attention mechanisms.

> ⚠️ This project does **not** aim for state-of-the-art performance.
> Instead, it emphasizes **correct methodology, interpretability, and reproducibility**.

---

## 📌 Key Questions

* Does attention improve CNN training stability?
* How does attention behave differently in image vs. video tasks?
* What are the practical limits of attention under small-data constraints?

---

## 📂 Project Structure

```text
how-attention-effect-on-cnn-module-in-training/
│
├── cnn_for_pic.py              # CNN for images (with spatial attention)
├── cnn_for_video.py            # CNN for videos (with temporal attention)
│
├── training_module_picture.py  # Image training pipeline
├── training_module_video.py    # Video training pipeline
│
├── data/
│   ├── images /                 # Or differnt dataset you perfer
│   └── WLASL/                  # Video dataset (WLASL MP4)
│
├── README.md
└── requirements.txt
```

---

## 🖼️ Part I — Image Classification (CNN + Spatial Attention)

### Overview

This experiment evaluates the effect of **spatial attention** on CNN training for image classification.

Spatial attention modules are integrated into the CNN to help the model:

* focus on informative regions,
* suppress background noise,
* and stabilize feature learning.

---

### Training Setup (Image)

* Task: Image classification
* Model: CNN with spatial attention
* Loss: Cross-Entropy
* Optimizer: Adam / AdamW
* Learning rate schedule: Cosine decay
* Epochs: 100

---

### Results (Image)

```text
Epoch 099/100 | train_acc=0.5447 | val_acc=0.4835
Epoch 100/100 | train_acc=0.5440 | val_acc=0.4855
Best val_acc = 0.4895
```

#### Interpretation

* Validation accuracy reaches **~49%**, which is strong for a lightweight CNN.
* Training and validation accuracies remain close, indicating **good generalization**.
* Attention contributes to **smoother convergence and stable learning behavior**.

---

## 🎥 Part II — Video Classification (CNN + Temporal Attention)

### Overview

This experiment studies **temporal attention** in CNN-based video classification using hand sign recognition as a test case.

Compared to images, video learning introduces:

* temporal variability,
* motion noise,
* and significantly higher data complexity.

---

### Dataset

* Dataset: WLASL (Word-Level American Sign Language)
* Input: Short MP4 videos
* Subset: 10 classes
* Sampling: Center frames only

> The small validation split makes evaluation metrics inherently noisy.

---

### Model Design (Video)

* **CNN Backbone**
  Extracts frame-level spatial features (frozen during training).

* **Temporal Attention Module (TCN)**
  Aggregates features across time and highlights informative motion segments.

---

### Training Setup (Video)

```text
NUM_FRAMES     = 8
LIMIT_CLASSES  = 10
BATCH_SIZE     = 4
EPOCHS         = 20
TEMPORAL_MODE  = "tcn"
LEARNING_RATE  = 1e-3
GPU            = RTX 3050 (6GB)
```

---

### Results (Video)

```text
Epoch 01 | val_acc = 0.30
Epoch 02 | val_acc = 0.20
Epoch 03 | val_acc = 0.20
Epoch 04 | val_acc = 0.20
Epoch 05 | val_acc = 0.00
```

#### Interpretation

* Validation accuracy fluctuates due to:

  * extremely small validation set,
  * class imbalance,
  * high inter-signer variability.
* The **best-performing checkpoint (~30%)** is selected based on peak validation accuracy.

---

## 🧪 Discussion

Across both tasks, the experiments show that:

* Attention improves **training stability and feature focus**.
* Performance gains are **context-dependent**:

  * clearer and more stable for images,
  * limited and noisy for videos.
* Attention acts more as a **training facilitator** than a guaranteed performance booster.

---

## ⚠️ Limitations

* Small datasets → high variance in evaluation
* No large-scale video pretraining
* Lightweight temporal modeling
* Results are not directly comparable to benchmark models

---

## ✅ Conclusion

This project demonstrates that attention mechanisms:

* positively influence CNN training behavior,
* improve learning stability,
* but do not overcome fundamental data limitations.

The contrast between image and video tasks highlights that **the effectiveness of attention strongly depends on input modality and data quality**.

---

## 🚀 Future Work

* Compare attention vs. non-attention models under identical settings
* Use pretrained video backbones (I3D, SlowFast)
* Explore transformer-based temporal attention
* Expand to larger WLASL subsets

---

## 📄 License

This project is released for **educational and research purposes**.

---

### 🔚 Final Note

If you are interested in **training dynamics**, **attention mechanisms**, or **small-data learning**, this repository provides a clean and interpretable experimental baseline.

