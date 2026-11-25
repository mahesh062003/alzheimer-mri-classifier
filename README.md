# Alzheimer's Disease Stage Classification from Brain MRI

A convolutional network that classifies brain MRI slices into four stages of
cognitive decline: **NonDemented**, **VeryMildDemented**, **MildDemented** and
**ModerateDemented**.

Staging is more useful than detection. A binary demented/not-demented model
tells a clinician little they do not already know; separating *very mild* from
*mild* is where an automated second read could inform follow-up intervals.

---

## Result

**94.15% test accuracy** across the four classes, after 30 epochs. Training
accuracy reached 100%.

**Limitation:** the split is image-level, not patient-level. Images are pooled
from the four class directories and divided with `train_test_split`, and the
source dataset is already augmented — so augmented variants of the same scan
can appear in both training and test sets. This inflates the test figure, and
the gap between 100% train and 94% test is consistent with it. Treat 94.15% as
an upper bound rather than an estimate of clinical generalisation; a
patient-level split would score lower.

---

## Model

```
Input 256 × 256 × 3
  Conv2D(64, 3×3) + ReLU  →  MaxPool 2×2
  Conv2D(32, 3×3) + ReLU  →  MaxPool 2×2
  Conv2D(32, 3×3) + ReLU  →  MaxPool 2×2
  Flatten
  Dense(16) + ReLU
  Dense(4)  + Softmax
```

Adam, sparse categorical cross-entropy, 30 epochs. A compact CNN trained from
scratch rather than a fine-tuned backbone.

---

## Layout

```
notebooks/Alzheimer.ipynb       Loading, model, training, evaluation
requirements.txt                Dependencies
```

## Trained weights

Not distributed. The notebook trains the model from scratch in 30 epochs — run
it against the dataset below to reproduce the result.

## Dataset

Kaggle **Alzheimer MRI (augmented)**. Not included — download it and place it
as `new_alzheimer_cnn/` with one subdirectory per class:

```
new_alzheimer_cnn/
  MildDemented/  ModerateDemented/  NonDemented/  VeryMildDemented/
```

The notebook's paths are `/content/...` as it was run in Colab; point them at
your local copy.

## Stack

Python · TensorFlow/Keras · scikit-learn · Pillow · NumPy
