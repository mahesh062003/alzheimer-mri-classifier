# Alzheimer's Disease Stage Classification from Brain MRI

A convolutional network that classifies brain MRI slices into four stages of
cognitive decline: **NonDemented**, **VeryMildDemented**, **MildDemented** and
**ModerateDemented**.

Staging matters more than detection here. A binary demented/not-demented model
says little a clinician does not already know; separating *very mild* from
*mild* is where an automated second read could actually inform follow-up
intervals.

---

## Result

**94.15% test accuracy** across the four classes (`cnn.evaluate` on the held-out
split), after 30 epochs. Training accuracy reached 100%.

### Read that number with the following caveat

The split is **image-level, not patient-level**. Images are pooled from the four
class directories and divided with `train_test_split`, and the source dataset is
already augmented — so augmented variants derived from the same original scan
can appear in both the training and test sets. That is leakage, and it inflates
the test figure.

The gap between 100% train and 94% test is consistent with it. A patient-level
split would almost certainly score lower, and **94.15% should be read as an
upper bound rather than an estimate of clinical generalisation.** Fixing this
properly means grouping by source scan before splitting, which the current
notebook does not do.

I am leaving the result as measured and stating the limitation, rather than
quietly reporting the higher number.

---

## Model

```
Input 256 x 256 x 3
  Conv2D(64, 3x3) + ReLU  ->  MaxPool 2x2
  Conv2D(32, 3x3) + ReLU  ->  MaxPool 2x2
  Conv2D(32, 3x3) + ReLU  ->  MaxPool 2x2
  Flatten
  Dense(16) + ReLU
  Dense(4)  + Softmax
```

Adam, sparse categorical cross-entropy, 30 epochs. Deliberately small — a
compact CNN trained from scratch, not a fine-tuned backbone.

---

## Layout

```
notebooks/Alzheimer.ipynb       Loading, model, training, evaluation
models/Alz_classifier.keras     Trained weights (Keras format)
models/Alz_classifier.h5        Same weights, HDF5
docs/AD detection.pdf           Project report
docs/dcmc report.pdf            Conference report
```

## Dataset

Kaggle **Alzheimer MRI (augmented)** — four class directories. Not included in
this repository; download it and place it as `new_alzheimer_cnn/` with one
subdirectory per class:

```
new_alzheimer_cnn/
  MildDemented/  ModerateDemented/  NonDemented/  VeryMildDemented/
```

The notebook's paths are `/content/...` as it was run in Google Colab — point
them at your local copy.

## Stack

Python · TensorFlow/Keras · scikit-learn · Pillow · NumPy
