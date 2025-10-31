# 🎥 VADL — Video Anomaly Detection & Localization

VADL is an AI-powered framework for detecting and localizing anomalies in video data.  
It leverages deep learning models to learn normal spatiotemporal patterns and identify deviations in unseen footage.

---

## 📁 Project Structure

```

VADL/
│
├── src/
│   ├── preprocessing.py     # Data preprocessing (frame extraction, normalization, etc.)
│   ├── train.py             # Model training
│   ├── evaluate.py          # Model evaluation on validation/test sets
│   ├── convertonnx.py       # Converts trained model to ONNX format for deployment
│   ├── infer.py             # Run inference on video streams or files
│   ├── model.py             # Model architecture
│   ├── dataset.py           # Dataset loader utilities
│   ├── helper.py            # Helper functions and metrics
│   └── config.py            # Configuration builder
│
├── data/                    # Raw and processed data
├── checkpoints/             # Saved model weights
├── outputs/                 # Evaluation results, ONNX models, etc.
├── requirements.txt
└── README.md

````

---

## ⚙️ Setup

### 1️⃣ Create Environment
```bash
python -m venv vadl-env
source vadl-env/bin/activate   # (Linux/Mac)
vadl-env\Scripts\activate      # (Windows)
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

Follow these steps **in order** to fully execute the VADL pipeline:

---

### 🧩 Step 1 — Preprocessing

Extract frames, normalize, and prepare datasets.

```bash
python src/preprocessing.py
```

This script will:

* Extract frames from input videos.
* Normalize and resize them.
* Split into training/validation/test sets.
* Save processed data under `data/processed/`.

---

### 🧠 Step 2 — Training

Train the anomaly detection model.

```bash
python src/train.py
```

This will:

* Load the preprocessed dataset.
* Initialize the model from `model.py`.
* Log metrics (accuracy, F1-score, loss) via Weights & Biases or local logs.
* Save the trained weights to `checkpoints/`.

---

### 📊 Step 3 — Evaluation

Evaluate the trained model on validation or test data.

```bash
python src/evaluate.py
```

Generates:

* Quantitative metrics (accuracy, F1, precision, recall).
* Visualization plots under `outputs/`.

---

### 🔄 Step 4 — Convert to ONNX

Convert the PyTorch model to ONNX format for deployment or inference optimization.

```bash
python src/convertonnx.py
```

Output:

```
outputs/model_vadl.onnx
```

---

### 🎯 Step 5 — Inference

Run inference on new video files or camera streams.

```bash
python src/infer.py
```

Example:

```bash
python src/infer.py --video path/to/test_video.mp4
```

Outputs:

* Frame-level anomaly scores.
* Optional visualization of detected anomalies (bounding boxes or heatmaps).
* Log file under `outputs/inference_results/`.

---

## 🧠 Key Features

* 🔍 **Anomaly Detection:** Learns normal spatiotemporal patterns and detects unusual activity.
* 🕵️ **Localization:** Highlights regions of anomalies in the video.
* ⚡ **ONNX Export:** Optimized model ready for real-time deployment.
* 📈 **Evaluation Metrics:** Includes F1, accuracy, and custom anomaly scoring.

---

## 🧩 Future Improvements

* Integrate real-time camera streaming with anomaly visualization.
* Add temporal consistency constraints for smoother detection.
* Implement multi-scale feature fusion for better localization accuracy.

---

## 🧾 Citation

If you use or refer to this project in your research, please cite:

```
@project{vadl2025,
  title={VADL: Video Anomaly Detection and Localization},
  author={Shreethar Raveenthar},
  year={2025},
  url={https://github.com/yourusername/VADL}
}
```

---

## 👨‍💻 Author

**Shreethar Raveenthar**
AI Researcher & Developer | Robotics & Deep Learning
📧 [your.email@example.com](mailto:your.email@example.com)
🌐 [GitHub Profile](https://github.com/yourusername)

---

## 🪪 License

This project is released under the **MIT License**.

```

---

Would you like me to make a **version with collapsible sections** (for example, click-to-expand “Setup” or “Running the Project” on GitHub)? It makes long READMEs like this much cleaner.
```
