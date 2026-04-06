# 🕵️ Deepfake Detector with Explainable AI (Grad-CAM)

An advanced Deep Learning system designed to identify and classify synthetically generated faces (Deepfakes). Engineered with **PyTorch** and accelerated via **Apple Silicon (MPS)**, this project integrates an autonomous face extraction engine and **Explainable AI (XAI)** to bring transparency to neural network decisions.

## 🚀 State-of-the-Art Performance
Trained locally over a massive **140,000 image dataset** (Kaggle's 'Real vs Fake Faces'), the model achieves robust generalization and industry-standard accuracy.

### 📊 Validation Results (20,000 Unseen Test Images)
- **General Accuracy:** **98.94%**
- **True Negatives (Accurately identified Real):** 9,896
- **True Positives (Accurately identified Fake):** 9,891
- **False Positives (Real flagged as Fake):** 104
- **False Negatives (Fake passed as Real):** 109

## 🧠 Core Architecture & Features
1. **Model Framework:** Custom-tailored **VGG16 Transfer Learning** architecture leveraging deep convolutional layers. A custom sequential classifier with robust Dropout (p=0.5) completely halts overfitting, converging `Validation Loss` to an outstanding `0.0354`.
2. **Autonomous Face Extraction (MTCNN):** Dynamic computer vision pipeline utilizing `facenet-pytorch`. Evaluates facial geometry boundaries:
   - **Narrow Shots:** Automatically bypasses double-cropping algorithms if the face covers > 80% geometry.
   - **Wide Shots:** Implements a strict `10 to 40 pixel margin` bounding box to extract specific biological failure points of Deepfakes (hairline, jaw contours, ears) without redundant background matrix. 
3. **Explainable AI (Grad-CAM):** To counteract the "Black-Box" nature of deep learning networks, the application generates a Heatmap over the evaluated faces. This maps exactly which microscopic artifacts (anomalous pixel behaviors) the VGG16 model focused on to make its binary decision.

## ⚙️ Installation & Usage
> **Security Notice:** The core 140K dataset and the serialized `.pth` deep weights (~520MB) are ignored from this repository to respect GitHub size limitations and data protection standardizations.

**Prerequisites:** Python 3.12+ (Virtual Environment is highly recommended).

```bash
# 1. Clone the repository
git clone https://github.com/rojhatkeles/Deepfake_Detector_Projesi.git
cd Deepfake_Detector_Projesi

# 2. Install Dependencies
pip install -r requirements.txt
pip install grad-cam

# 3. Launch the Local User Interface
streamlit run ui/app.py
```

## 🛠 Tech Stack
*   **Deep Learning Engine**: PyTorch, Torchvision
*   **Computer Vision**: MTCNN, OpenCV, Pillow
*   **XAI Toolkit**: pytorch-grad-cam
*   **Data Science**: Numpy, Pandas, TQDM
*   **UI / Dashboard**: Streamlit
*   **Hardware Acceleration**: MPS (Metal Performance Shaders) natively mapped.
