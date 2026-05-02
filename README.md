# 🦺 AI-Based Workplace Safety — Real-Time PPE Detection

A real-time Personal Protective Equipment (PPE) detection system built on **YOLOv8** and **OpenCV**, capable of identifying PPE compliance and violations from live webcam feeds or static images. Developed as a B.Tech Minor Project at **Jaypee University of Engineering and Technology, Guna**.

---

## 👥 Team

| Name | Enrollment No. |
|------|----------------|
| Shejal Dubey | 231B309 |
| Shivansh | 231B317 |
| Shreyansh Mishra | 231B324 |

**Guided by:** Prof. Vipin Tyagi, Dean (Academic & Research), JUET

---

## 📌 Overview

Manual PPE monitoring is slow, resource-intensive, and prone to human error. This project replaces manual supervision with an **automated, real-time computer vision pipeline** that:

- Detects workers wearing or not wearing required PPE
- Classifies each frame as `SAFE` or `UNSAFE` in real time
- Displays color-coded bounding boxes and violation alert banners
- Runs locally with no cloud dependency

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| Precision | **92.57%** |
| Recall | **82.15%** |
| mAP@0.5 | **87.37%** |
| Training Platform | Kaggle GPU (NVIDIA P100 / T4) |
| Model Format | `best.pt` (PyTorch weights) |

---

## 🗂️ Detection Classes

The model detects **10 classes** from the Construction Site Safety Image Dataset:

| Class | Type |
|-------|------|
| `Hardhat` | ✅ Safe |
| `Mask` | ✅ Safe |
| `Safety Vest` | ✅ Safe |
| `NO-Hardhat` | ❌ Violation |
| `NO-Mask` | ❌ Violation |
| `NO-Safety Vest` | ❌ Violation |
| `Person` | 🔵 Neutral |
| `Safety Cone` | 🔵 Neutral |
| `Machinery` | 🔵 Neutral |
| `Vehicle` | 🔵 Neutral |

---

## 🧠 Tech Stack

- **Model:** YOLOv8 (Ultralytics)
- **Video Processing:** OpenCV (`cv2`)
- **Deep Learning Framework:** PyTorch
- **Training Platform:** Kaggle Notebooks (GPU)
- **Dataset:** [Construction Site Safety Image Dataset — Roboflow](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
- **Language:** Python 3.8+

---

## 📁 Project Structure

```
ppe-detection/
├── best.pt                   # Trained YOLOv8 model weights
├── detect_webcam.py          # Main real-time detection script
├── notebook87bc419a18.ipynb  # Kaggle training notebook
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/ppe-detection.git
cd ppe-detection
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

> Requires Python 3.8+. A GPU is strongly recommended for real-time performance.

**3. Run the detection demo**

```bash
python detect_webcam.py
```

Press **`Q`** to exit the webcam feed.

---

## 🔧 Configuration

You can tune the following constants at the top of `detect_webcam.py`:

```python
MODEL_PATH   = 'best.pt'   # Path to model weights
CONFIDENCE   = 0.60        # Detection confidence threshold
WEBCAM_INDEX = 0           # Webcam device index (0 = default camera)
```

---

## 🚀 How It Works

The system runs a sequential 6-stage pipeline on every captured frame:

```
Input Capture → YOLO Inference → Result Extraction → Safety Classification → Bounding Box Rendering → Alert Display
```

1. **Input Capture** — Reads frames from webcam via OpenCV `VideoCapture`
2. **YOLO Inference** — Runs `best.pt` on each frame using the Ultralytics API
3. **Result Extraction** — Parses bounding boxes, class labels, and confidence scores
4. **Safety Classification** — Flags the frame as `UNSAFE` if any `NO-*` class is detected
5. **Bounding Box Rendering** — Draws color-coded boxes (🟢 safe / 🔴 violation) with confidence scores
6. **Alert Display** — Renders a prominent violation banner at the top of the frame when unsafe

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Total Images | 2,687 |
| Training Images | 2,605 |
| Test Images | 82 |
| Annotation Format | YOLO-format (normalized bounding boxes in `.txt` files) |
| Source | Roboflow / Kaggle |

---

## 📉 Training

Training was performed on Kaggle's free GPU environment using the Ultralytics YOLOv8 training API:

- **Base Model:** YOLOv8n / YOLOv8s
- **Input Size:** 640×640 px
- **Epochs:** 100
- **Checkpointing:** Best model saved as `best.pt` based on validation mAP@0.5

Training metrics (box loss, cls loss, DFL loss) converge smoothly across epochs. See the Kaggle training notebook (`notebook87bc419a18.ipynb`) for full loss curves and evaluation plots.

---

## ⚠️ Known Limitations

- Detection accuracy degrades on distant or small objects
- Reduced FPS on CPU-only machines — a GPU is recommended for smooth real-time output
- Dataset consists primarily of daytime images; low-light and adverse weather performance is limited
- No per-worker identity tracking across frames
- Currently supports a single camera feed only

---

## 🔭 Future Work

- Multi-camera CCTV network integration
- Worker identity tracking across frames (DeepSORT / ByteTrack)
- Extended PPE class coverage: goggles, gloves, safety boots
- Automated SMS/email/alarm alerts on violation detection
- Edge deployment on NVIDIA Jetson for on-site use
- Low-light preprocessing (histogram equalization, adaptive illumination normalization)
- Docker containerization for portable deployment

---

## 📄 License

This project was developed for academic purposes as a B.Tech Minor Project at JUET Guna (May 2026). Not intended for commercial use.

---

## 📚 References

- Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," CVPR 2016
- Ultralytics YOLOv8 Documentation — https://docs.ultralytics.com
- Roboflow Construction Site Safety Dataset — https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
- OpenCV — https://opencv.org
- PyTorch — https://pytorch.org
- International Labour Organization Safety Report — https://www.ilo.org
