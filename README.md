# 🕵️ LensAI Security: Enterprise Deepfake Detector

An advanced B2B-grade Deep Learning system designed to identify and classify synthetically generated faces (Deepfakes) in real-time. Engineered with **PyTorch** and deployed via a dynamic **FastAPI + React.js** microservices architecture. It integrates an autonomous face extraction engine and **Explainable AI (XAI)** to bring transparency to neural network decisions.

## 🚀 State-of-the-Art Performance
Trained locally over a massive **140,000 image dataset** (Kaggle's 'Real vs Fake Faces' array), the VGG-16 model achieves robust generalization and industry-standard accuracy.

### 📊 Validation Results (20,000 Unseen Test Images)
- **Epoch Performance:** `Validation Loss: 0.0354`
- **General Accuracy:** **98.94%**
- **True Negatives (Accurately identified Real):** 9,896
- **True Positives (Accurately identified Fake):** 9,891
- **False Positives (Real flagged as Fake):** 104
- **False Negatives (Fake passed as Real):** 109

## 🧠 Core Architecture & Features
1. **Model Framework:** Custom-tailored **VGG16 Transfer Learning** architecture leveraging deep convolutional matrices.
2. **Autonomous Face Extraction (MTCNN / 80% Threshold):** Dynamic computer vision pipeline utilizing `facenet-pytorch`. Evaluates facial geometry boundaries and extracts optimal matrices while mitigating double-cropping phenomena on extreme close-ups.
3. **Explainable AI (Grad-CAM Base64 Engine):** Converts "Black-Box" ML evaluations into actionable intelligence. The API transmits heatmaps tracing microscopic gradients via Base64 back to the React UI.
4. **Decoupled Application Layer (SaaS Ready):** Transitioned from standard monocodes to an enterprise-grade `React.js` (Frontend, Port 8090) and `FastAPI` (Backend, Port 8089) configuration. 

## ⚙️ Detailed Installation & Build Process

> ⚠️ **IMPORTANT NOTICE FOR JURY & EVALUATORS:**
> Due to GitHub's strict file size protocols, the compiled deep learning weights (`best_model.pth`, ~520MB) cannot be pushed to this repository. 
> To successfully execute the AI predictions locally, please **[DOWNLOAD THE MODEL WEIGHTS HERE](#)** *(Insert Google Drive Link Here)* and place the `best_model.pth` file directly into the `saved_models/` directory before launching the app.

**Prerequisites:** 
- Python 3.10+
- Node.js 18+ and npm
- Deep Learning weights file (`best_model.pth`) strictly placed inside `saved_models/`.

### Step-by-Step Deployment Guide

**1. Clone the repository**
Download the application files to your local environment.
```bash
git clone https://github.com/rojhatkeles/Deepfake_Detector_Projesi.git
cd Deepfake_Detector_Projesi
```

**2. Setup Python Virtual Environment (Backend)**
It is strictly required to run the AI network in an isolated virtual environment to prevent package collisions.
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
# For macOS/Linux:
source .venv/bin/activate
# For Windows:
# .venv\Scripts\activate
```

**3. Install AI & Backend Server Dependencies**
With the virtual environment active (`(.venv)` should be visible in your terminal), install the PyTorch frameworks and FastAPI server modules.
```bash
pip install -r requirements.txt
```

**4. Install Node.js Dependencies (Frontend)**
Navigate to the React application directory and download the client-side packages.
```bash
cd frontend
npm install
cd ..
```

**5. Autonomous System Boot**
A unified orchestrator script (`run.sh`) is provided to securely orchestrate ports (8089 for API, 8090 for UI), wipe any ghost deployments, and synchronously boot both layers of the architecture.

```bash
# Grant execution rights to the script
chmod +x run.sh

# Fire up the Architectural Stack
./run.sh
```
*The script will bind the frontend and automatically proxy you to the modern React App at `http://localhost:8090`.*

---

## 🛠 Tech Stack
*   **Deep Learning Backend**: PyTorch, Torchvision, FastAPI, Uvicorn
*   **Computer Vision**: MTCNN, OpenCV, Pillow, Grad-CAM XAI
*   **Reactivity / UI**: Node.js, React, Vite, Vanilla Modern CSS (Inter/Outfit UI)
*   **Execution Infrastructure**: Bash scripting, Apple Silicon (MPS) accelerated.
