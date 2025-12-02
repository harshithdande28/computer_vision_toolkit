# Camera Calibration & Computer Vision App

A comprehensive Flask-based web application covering multiple computer vision modules including camera calibration, image processing, object tracking, and AI-based segmentation.

## 🎯 Features

| Module       | Description                                         |
| ------------ | --------------------------------------------------- |
| **Module 1** | Camera Calibration & Perspective Projection         |
| **Module 2** | Gaussian Blur & Fourier Transform                   |
| **Module 3** | Edge Detection, Corner Detection, Template Matching |
| **Module 4** | Image Stitching (Panorama) & SIFT Feature Matching  |
| **Module 5** | Object Tracking (Marker, Markerless, SAM2)          |
| **Module 6** | ArUco-Based Object Segmentation                     |
| **Module 7** | Stereo Vision & Size Estimation                     |
| **Module 8** | Real-Time Pose & Hand Tracking (MediaPipe)          |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11** (Required for MediaPipe and SAM2 compatibility)
- Git
- Webcam (for real-time tracking modules)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/camera_calibration_app.git
cd camera_calibration_app

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download SAM2 checkpoint (for Module 5 & 6)
# Create folder and download:
mkdir sam2_checkpoints
# Download from: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
# Place in sam2_checkpoints/

# 6. Run the application
python app.py
```

### Open in Browser

```
http://127.0.0.1:5000
```

---

## 📦 Dependencies

### Core Requirements

```
flask
opencv-python
opencv-contrib-python
numpy
scipy
```

### For Pose & Hand Tracking (Module 8)

```
mediapipe
```

### For SAM2 Segmentation (Module 5 & 6)

```
torch
torchvision
segment-anything-2  # Install from: pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Install All at Once

```bash
pip install flask opencv-python opencv-contrib-python numpy scipy mediapipe
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

---

## 📁 Project Structure

```
camera_calibration_app/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── templates/                  # HTML templates
│   ├── home.html
│   ├── perspective_projection.html
│   ├── gaussian_blur_nd_fourier_transform.html
│   ├── template_matching.html
│   ├── gradient_and_log.html
│   ├── double_threshold_edges.html
│   ├── manual_corners.html
│   ├── boundary_detection.html
│   ├── aruco_segmentation.html
│   ├── image_stitching.html
│   ├── sift_feature_matching.html
│   ├── module5.html
│   ├── module5_marker.html
│   ├── module5_markerless.html
│   ├── module5_sam2.html
│   ├── stereo_size_estimation.html
│   ├── stereo_calibration.html
│   └── pose_hand_tracking.html
├── static/
│   └── sam2/
│       ├── car_video.mp4       # Sample video for SAM2
│       └── masks.npz           # Generated masks
├── uploads/                    # User uploaded files
├── sam2_checkpoints/           # SAM2 model weights
│   └── sam2_hiera_tiny.pt
└── generate_sam2_masks_bbox.py # SAM2 mask generator script
```

---

## 🔧 Module Details

### Module 1: Camera Calibration

- Upload chessboard images (7x6 pattern)
- Extracts intrinsic parameters (Fx, Fy, Ox, Oy)
- Perspective projection demo

### Module 2: Gaussian Blur & Fourier Transform

- Applies Gaussian blur to images
- Performs Wiener deconvolution to restore images
- Visualizes frequency domain

### Module 3: Image Processing

- **Gradient & LoG**: Sobel gradients, Laplacian of Gaussian
- **Edge Detection**: Manual double threshold edge detection
- **Corner Detection**: Harris corner detector (manual implementation)
- **Template Matching**: Find and blur matching regions
- **Boundary Detection**: Interactive ROI-based boundary detection

### Module 4: Feature Matching & Stitching

- **Image Stitching**: Create panoramas from multiple images
- **SIFT Matching**: From-scratch SIFT with RANSAC homography

### Module 5: Object Tracking

- **Marker-based**: ArUco marker detection and tracking
- **Markerless**: CSRT tracker with ROI selection
- **SAM2**: AI-based video object segmentation

### Module 6: ArUco Segmentation

- Non-rectangular object segmentation
- Place ArUco markers on object boundary
- Compares with SAM2 segmentation (IoU score)

### Module 7: Stereo Vision

- Stereo camera calibration
- Disparity map generation
- 3D object size estimation
- Real-time body pose detection (33 landmarks)
- Hand tracking (21 landmarks per hand)
- CSV data export for analysis

---

## 🎥 Sample Data

### For SAM2 (Module 5)

Place a video file at: `static/sam2/car_video.mp4`

### For ArUco Segmentation (Module 6)

- Print ArUco markers (DICT_4X4_50, IDs 0-49)
- Place markers around object boundary
- Capture 10+ images from various angles

### For Stereo Calibration (Module 7)

- Capture stereo image pairs with chessboard pattern
- Upload left and right images separately

---

## ⚠️ Troubleshooting

### MediaPipe not working

```bash
# Ensure Python 3.11 (not 3.13)
python --version  # Should show 3.11.x
pip install mediapipe
```

### SAM2 not loading

```bash
# Verify checkpoint exists
dir sam2_checkpoints\sam2_hiera_tiny.pt

# Reinstall SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Webcam not detected

- Check camera permissions in Windows Settings
- Try a different USB port
- Verify with: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

---

## 📝 License

This project is for educational purposes.

---

## 👨‍💻 Author

Lourdu Gnana Harshith Dande
