# VADL — Video Anomaly Detection & Localization

VADL is an AI-powered framework for detecting and localizing anomalies in video data.  
It leverages deep learning models to learn normal spatiotemporal patterns and identify deviations in unseen footage.

---

## 📁 Project Structure

```

VADL/
│
├── src/
│   ├── results/             # Contains image output and model weight output
│   │   ├── images/          # Contains image output of Evaluation and EDA
│   │   └── model_files/     # Contains the PyTorch model weight & the ONNX output
│   ├── preprocessing.py     # Data preprocessing (frame extraction, normalization, etc.)
│   ├── train.py             # Model training
│   ├── evaluate.py          # Model evaluation on validation/test sets
│   ├── convertonnx.py       # Converts trained model to ONNX format for deployment
│   ├── infer.py             # Run inference on video streams or files
│   ├── model.py             # Model architecture
│   ├── dataset.py           # Dataset loader utilities
│   ├── helper.py            # Helper functions and metrics
│   └── config.py            # Configuration builder
├── requirements.txt
└── README.md

````

---

## ⚙️ Setup

### Create Environment
```bash
python -m venv vadl-env
source vadl-env/bin/activate   # (Linux/Mac)
vadl-env\Scripts\activate      # (Windows)
````

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Follow these steps **in order** to fully execute the VADL pipeline:

---

### Step 1 — Preprocessing

Extract frames, normalize, and prepare datasets.

```bash
python -m src.preprocessing
```

This script will:

* Extract frames from input videos.
* Sample one out of 16 frames.
* Sample until it reaches the end of video, or 256.

---

### Step 2 — Training

Train the anomaly detection model.

```bash
python -m src.train
```

This will:

* Load the preprocessed dataset.
* Initialize the model from `model.py`.
* Log metrics (accuracy, F1-score, loss) via Weights & Biases or local logs.
* Save the trained weights to `src/results/model_files/`.

---

### Step 3 — Evaluation

Evaluate the trained model on validation and test data.

```bash
python -m src.evaluate
```

Generates:

* Quantitative metrics (accuracy, F1, precision, recall).
* Visualization plots under `src/results/images/`.

---

### Step 4 — Convert to ONNX

Convert the PyTorch model to ONNX format for deployment or inference optimization.

```bash
python -m src.convertonnx
```

Output:

```
src/model_files/VADL_model_ONNX.onnx
```

---

### Step 5 — Inference

Run inference on new video files or camera streams.

```bash
python -m src.infer
```


Outputs:

* Frame-level anomaly scores.
* Bounding box visualization of detected anomalies.

---

# PLOVAD Localization & Optimization Report

This repository contains the integration of localization in PLOVAD architecture and performance analysis. The objective was to include localization head (Output the bounding box wherever an anomaly is occuring), and minimize the inference time.
The model was inferenced on NVIDIA A4000 with 16GB VRAM.

We modified the original PLOVAD model by removing the multi-class classifier and adding a spatial feature map. The spatial feature map is then downsampled into 1 channel to be a heatmap. The model learns that high heatmap value is anomalous and low heatmap value is normal.

![Modified PLOVAD Architecture](assets/6071115128679107349.jpg)

---
After further analyzing the heatmap values across all 3 data sets, and 2 labels, we have decided for using **0.29** as a threshold for the heatmap. If the heatmap value is below the threshold, it's considered normal, but if it's over the threshold, its considered anomalous.

Here are the plots to support the threshold hyperparameter setting. We are using the normal label from the validation dataset to get the threshold.

![Histogram](src/results/images/Histogram.png)
![Violinplot](src/results/images/Violinplot.png)

---
After including the localization head, our next goal is to proceed with exporting the PyTorch model to ONNX.

Here is the time taken to do inference for video samples of different frame numbers.

## ONNX vs PyTorch
| Frames | ONNX | PyTorch |
|--------|---------|------|
| 106 | 1.64 | 1.07 |
| 256 | 1.65 | 2.57 |
| 256 | 1.71 | 2.51 |
| 256 | 1.50 | 2.53 |
| 256 | 1.71 | 2.53 |
| 177 | 1.73 | 1.75 |
| 72 | 1.87 | 0.68 |
| 72 | 2.12 | 0.68 |
| 109 | 1.76 | 1.12 |
| 165 | 1.79 | 1.73 |
| 130 | 1.66 | 1.32 |
| 196 | 1.90 | 1.96 |

> Observation: We can see that no matter the frame number, the time taken is always around 1.6s to 1.8s for ONNX while for PyTorch it depends on the number of frames.
> If the number of frames is high, then ONNX does inference faster and if the number of frame is low, then PyTorch does inference faster.
> In our case, we will be setting the frame number at 256 for inferencing. It takes ONNX around 1.7 second and PyTorch around 2.6 seconds. That's a **38%** speed increase.
