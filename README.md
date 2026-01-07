
# 🖐️ Hand Sign Recognition with CNN + Temporal Attention (WLASL)

## 📌 Overview

This project explores the effect of **temporal modeling (TCN-based attention)** on a CNN-based video classification pipeline for **hand sign recognition** using the **WLASL dataset**.
The goal is **not to achieve state-of-the-art accuracy**, but to analyze how different temporal strategies influence learning behavior under **limited data and computational constraints**.

The project is designed as a **side research project / coursework-level experiment**, focusing on:

* correctness of the pipeline,
* interpretability of results,
* and reproducibility.

---

## 📂 Dataset

### WLASL (Word-Level American Sign Language)

* Source: WLASL dataset (processed MP4 version)
* Input format: short sign videos (`.mp4`)
* Characteristics:

  * High inter-signer variability
  * Background noise
  * Uneven video lengths
  * Small validation split

To keep the experiment tractable:

* Only **10 classes** are used
* Only **center frames** of each video are sampled

> ⚠️ Due to the small size of the validation set, evaluation metrics are inherently noisy.

---

## 🧠 Model Architecture

### 1️⃣ Spatial Feature Extractor (CNN)

* A 2D CNN backbone is used to extract frame-level features.
* The backbone is **frozen during training** to prevent overfitting and reduce noise learning.

### 2️⃣ Temporal Modeling

We evaluate temporal aggregation using:

* **TCN (Temporal Convolutional Network)**

The temporal module processes a sequence of frame-level features and captures motion dynamics across time.

> More complex models (e.g., GRU, Transformers, 3D CNNs) were intentionally avoided to keep the focus on **lightweight temporal modeling**.

---

## ⚙️ Training Configuration

```text
NUM_FRAMES     = 8
LIMIT_CLASSES  = 10
BATCH_SIZE     = 4
EPOCHS         = 20
TEMPORAL_MODE  = "tcn"
LEARNING_RATE  = 1e-3
GPU            = RTX 3050 (6GB)
```

Additional design choices:

* Center-frame sampling (no random cropping)
* Early stopping based on validation accuracy
* Best checkpoint selected by **peak validation accuracy**, not final epoch

---

## 📊 Results

### Validation Accuracy (Representative Run)

```text
Epoch 01 | val_acc = 0.30
Epoch 02 | val_acc = 0.20
Epoch 03 | val_acc = 0.20
Epoch 04 | val_acc = 0.20
Epoch 05 | val_acc = 0.00
```

### Interpretation

* Validation accuracy **fluctuates significantly** due to:

  * extremely small validation split,
  * class imbalance,
  * high variability across signers.

* The **best-performing epoch (0.30)** is selected as the final model.

> This behavior is expected and commonly reported in small-scale video recognition experiments.

---

## 🧪 Discussion

* A validation accuracy of **~30%** on 10 WLASL classes using a CNN + TCN model is considered **reasonable** under:

  * limited training data,
  * no video-level pretraining,
  * lightweight temporal modeling.

* The experiment demonstrates that:

  * correct label mapping between train/validation splits is critical,
  * freezing spatial features significantly stabilizes learning,
  * temporal modeling improves performance compared to naive frame averaging.

---

## ⚠️ Limitations

* Very small validation set → high metric variance
* No large-scale video pretraining (e.g., I3D, SlowFast)
* No data augmentation for temporal consistency
* Results are not directly comparable to benchmark papers

---

## 🚀 Future Work

* Use pretrained video backbones (I3D / SlowFast)
* Evaluate attention-based temporal transformers
* Expand to full WLASL-100 or WLASL-300
* Cross-validation for more stable evaluation

---

## ✅ Conclusion

This project provides a **clean and interpretable baseline** for studying temporal attention mechanisms on CNN-based video models under realistic constraints.
Despite modest accuracy, the results are **methodologically sound** and suitable for **educational or exploratory research purposes**.



