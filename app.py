from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2 as cv
import os
import base64
import cv2.aruco as aruco
from flask import Response
from flask import request, jsonify
import subprocess
import sys
import os
from flask import request, render_template
from werkzeug.utils import secure_filename

# ============= Cloud Deployment Detection =============
# Detect if running on cloud (Render, Heroku, Railway, etc.)
IS_CLOUD = bool(os.environ.get('RENDER') or 
                os.environ.get('DYNO') or 
                os.environ.get('RAILWAY_ENVIRONMENT') or
                os.environ.get('PORT'))
if IS_CLOUD:
    print("☁️  Running in CLOUD mode - webcam features disabled")
else:
    print("💻 Running in LOCAL mode - all features available")
try:
    import mediapipe as mp  # type: ignore
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print("WARNING: MediaPipe not available. Pose/Hand tracking features disabled.")
    print("MediaPipe requires Python 3.8-3.12. You are using Python 3.13.")

# SAM2 initialization
SAM2_AVAILABLE = False
sam2_predictor = None
try:
    import torch  # type: ignore
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    
    SAM2_CHECKPOINT = os.path.join(os.path.dirname(__file__), "sam2_checkpoints", "sam2_hiera_tiny.pt")
    SAM2_CONFIG = "configs/sam2/sam2_hiera_t.yaml"
    
    if os.path.exists(SAM2_CHECKPOINT):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM2 on {device}...")
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        SAM2_AVAILABLE = True
        print("✅ SAM2 loaded successfully!")
    else:
        print(f"⚠️ SAM2 checkpoint not found at: {SAM2_CHECKPOINT}")
except ImportError as e:
    print(f"⚠️ SAM2 not available: {e}")
except Exception as e:
    print(f"⚠️ SAM2 loading failed: {e}")

if not SAM2_AVAILABLE:
    print("📢 Using GrabCut fallback for segmentation.")

import csv
from datetime import datetime


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chessboard settings (7x6 corners)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# ----------------------------
# Global variables to store intrinsic parameters
# ----------------------------
calibration_done = False
Fx, Fy, Ox, Oy = None, None, None, None

# this is the home page
@app.route("/")
def index():
    return render_template("home.html")

# API to check deployment status
@app.route("/api/status")
def api_status():
    return jsonify({
        "is_cloud": IS_CLOUD,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "sam2_available": SAM2_AVAILABLE,
        "webcam_available": not IS_CLOUD,
        "message": "Cloud deployment - some features disabled" if IS_CLOUD else "Local deployment - all features available"
    })

# added a new route for prespective projection
@app.route("/perspective_projection")
def perspective_projection():
    return render_template("perspective_projection.html")

@app.route("/calibrate", methods=["POST"])
def calibrate():
    global calibration_done, Fx, Fy, Ox, Oy
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    objpoints, imgpoints = [], []
    gray = None

    for f in files:
        filepath = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filepath)
        img = cv.imread(filepath)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    if not objpoints:
        return jsonify({"error": "No chessboard corners detected"}), 400

    # Run calibration (ignore distortion coefficients)
    ret, mtx, _, _, _ = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Extract intrinsic parameters
    Fx, Fy = mtx[0, 0], mtx[1, 1]
    Ox, Oy = mtx[0, 2], mtx[1, 2]

    calibration_done = True

    return jsonify({
        "Fx": float(Fx),
        "Fy": float(Fy),
        "Ox": float(Ox),
        "Oy": float(Oy)
    })

@app.route("/get_intrinsics", methods=["GET"])
def get_intrinsics():
    if not calibration_done:
        return jsonify({"error": "Calibration not done yet"}), 400

    return jsonify({
        "Fx": float(Fx),
        "Fy": float(Fy),
        "Ox": float(Ox),
        "Oy": float(Oy)
    })

# Fourier Transform Page
# -------------------------------
@app.route("/gaussian_blur_nd_fourier_transform")
def gaussian_blur_nd_fourier_transform():
    return render_template("gaussian_blur_nd_fourier_transform.html")

# Gaussian blurring and Fourier Transform Processing API
@app.route("/process_fourier", methods=["POST"])
def process_fourier():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # --- Configuration ---
    GAUSSIAN_KERNEL_SIZE = (15, 15)
    GAUSSIAN_SIGMA = 1.5
    WIENER_K = 0.001

    # --- Step 1: Load Image ---
    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    L = img.astype(np.float32) / 255.0
    rows, cols = L.shape

    # --- Step 2: Apply Gaussian Blur ---
    L_b = cv.GaussianBlur(L, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)

    # --- Step 3: Construct Gaussian Kernel ---
    G_kernel = cv.getGaussianKernel(GAUSSIAN_KERNEL_SIZE[0], GAUSSIAN_SIGMA)
    G_kernel = G_kernel @ G_kernel.T

    # Pad kernel to image size
    G_padded = np.zeros((rows, cols), dtype=np.float32)
    h_k, w_k = G_kernel.shape
    r_start, c_start = (rows - h_k) // 2, (cols - w_k) // 2
    G_padded[r_start:r_start + h_k, c_start:c_start + w_k] = G_kernel

    # Shift zero-frequency component to center
    G_padded = np.fft.ifftshift(G_padded)

    # --- Step 4: Fourier Transform ---
    F_G = np.fft.fft2(G_padded)
    F_Lb = np.fft.fft2(L_b)

    # --- Step 5: Wiener Deconvolution ---
    F_G_conj = np.conjugate(F_G)
    F_G_abs_sq = np.abs(F_G) ** 2
    H_w = F_G_conj / (F_G_abs_sq + WIENER_K)
    F_L_hat = H_w * F_Lb

    # --- Step 6: Inverse Transform ---
    L_hat = np.fft.ifft2(F_L_hat)
    L_hat = np.abs(L_hat)

    # --- Step 7: Convert for Display ---
    L_display = (np.clip(L, 0, 1) * 255).astype(np.uint8)
    L_b_display = (np.clip(L_b, 0, 1) * 255).astype(np.uint8)
    L_hat_display = (np.clip(L_hat, 0, 1) * 255).astype(np.uint8)

    # --- Helper: Convert to Base64 ---
    def to_base64(img_array):
        _, buf = cv.imencode(".png", img_array)
        return base64.b64encode(buf).decode("utf-8")

    # --- Step 8: Return JSON with Encoded Images ---
    return jsonify({
        "original": to_base64(L_display),
        "blurred": to_base64(L_b_display),
        "restored": to_base64(L_hat_display),
        "info": {
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "kernel_size": GAUSSIAN_KERNEL_SIZE,
            "wiener_k": WIENER_K
        }
    })


# ==============================================================
# TEMPLATE MATCHING PAGE
# ==============================================================
@app.route("/template_matching")
def template_matching():
    return render_template("template_matching.html")


# ==============================================================
# TEMPLATE MATCHING PROCESSING API
# ==============================================================
@app.route("/process_template_matching", methods=["POST"])
def process_template_matching():
    main_file = request.files.get("main_image")
    template_files = request.files.getlist("template_images")

    if not main_file or not template_files:
        return jsonify({"error": "Please upload one main image and at least one template image."}), 400

    main_path = os.path.join(UPLOAD_FOLDER, main_file.filename)
    main_file.save(main_path)
    img_main = cv.imread(main_path, cv.IMREAD_GRAYSCALE)
    assert img_main is not None, "Main image could not be read. Check the file path."
    img_original = img_main.copy()

    results = []
    methods = ['TM_CCOEFF']

    for idx, temp in enumerate(template_files, start=1):
        template_path = os.path.join(UPLOAD_FOLDER, temp.filename)
        temp.save(template_path)

        template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
        if template is None:
            print(f"[Warning] Template {temp.filename} not found — skipping.")
            continue

        w, h = template.shape[::-1]

        for meth in methods:
            img = img_original.copy()
            method = getattr(cv, meth)
            res = cv.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

            # Select best match
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # --- Blur the detected region ---
            roi = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            if roi.size > 0:
                blurred_roi = cv.GaussianBlur(roi, (21, 21), 0)
                img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blurred_roi

            # --- Draw rectangle after blurring ---
            cv.rectangle(img, top_left, bottom_right, (255, 255, 255), 2)

            # --- Encode for JSON output ---
            _, buf_main = cv.imencode(".png", img_main)
            _, buf_template = cv.imencode(".png", template)
            _, buf_result = cv.imencode(".png", img)

            results.append({
                "template_name": temp.filename,
                "main": base64.b64encode(buf_main).decode("utf-8"),
                "template": base64.b64encode(buf_template).decode("utf-8"),
                "result": base64.b64encode(buf_result).decode("utf-8")
            })

    return jsonify(results)


# ==============================================================
# MODULE 3: GRADIENT IMAGES AND LAPLACIAN OF GAUSSIAN
# ==============================================================
@app.route("/gradient_and_log")
def gradient_and_log():
    return render_template("gradient_and_log.html")

@app.route("/process_gradient_log", methods=["POST"])
def process_gradient_log():
    import base64
    files = request.files.getlist("images")  # ⬅️ getlist for multiple files
    if not files:
        return jsonify({"error": "No image uploaded"}), 400

    results = []

    for f in files:
        filepath = os.path.join("uploads", f.filename)
        f.save(filepath)

        img = cv.imread(filepath)
        if img is None:
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
        magnitude = cv.magnitude(grad_x, grad_y)
        angle = cv.phase(grad_x, grad_y, angleInDegrees=True)
        mag_vis = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        ang_vis = cv.normalize(angle, None, 0, 255, cv.NORM_MINMAX)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        log = cv.Laplacian(blurred, cv.CV_64F)
        log_vis = cv.normalize(log, None, 0, 255, cv.NORM_MINMAX)

        def to_base64(img):
            _, buf = cv.imencode(".png", img)
            return base64.b64encode(buf).decode("utf-8")

        results.append({
            "filename": f.filename,
            "original": to_base64(img),
            "magnitude": to_base64(mag_vis),
            "angle": to_base64(ang_vis),
            "log": to_base64(log_vis)
        })

    return jsonify({"processed_images": results})

# ==============================================================
# MODULE 3 — MANUAL EDGE DETECTION (DOUBLE THRESHOLD)
# ==============================================================
@app.route("/double_threshold_edges")
def double_threshold_edges():
    return render_template("double_threshold_edges.html")

@app.route("/process_double_threshold", methods=["POST"])
def process_double_threshold():
    import base64, math

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    results = []
    for f in files:
        filepath = os.path.join("uploads", f.filename)
        f.save(filepath)

        # ---- Step 1: Load grayscale ----
        img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        gradient_mag = np.zeros((H, W), dtype=np.float32)

        # ---- Step 2: Compute gradients manually ----
        for i in range(1, H-1):
            for j in range(1, W-1):
                gx = int(img[i, j+1]) - int(img[i, j-1])
                gy = int(img[i+1, j]) - int(img[i-1, j])
                gradient_mag[i, j] = math.sqrt(gx**2 + gy**2)

        # ---- Step 3: Thresholds ----
        mean_val = np.mean(gradient_mag)
        std_val = np.std(gradient_mag)
        T_high = mean_val + std_val
        T_low = 0.5 * T_high

        # ---- Step 4: Edge classification ----
        strong_edges = np.zeros_like(img, dtype=np.uint8)
        weak_edges = np.zeros_like(img, dtype=np.uint8)
        final_edges = np.zeros_like(img, dtype=np.uint8)

        for i in range(1, H-1):
            for j in range(1, W-1):
                val = gradient_mag[i, j]
                if val >= T_high:
                    strong_edges[i, j] = 255
                elif val >= T_low:
                    weak_edges[i, j] = 75

        # ---- Step 5: Edge linking ----
        for i in range(1, H-1):
            for j in range(1, W-1):
                if weak_edges[i, j] == 75:
                    neighbors = strong_edges[i-1:i+2, j-1:j+2]
                    if np.any(neighbors == 255):
                        final_edges[i, j] = 255
                elif strong_edges[i, j] == 255:
                    final_edges[i, j] = 255

        # ---- Base64 encode all outputs ----
        def to_base64(img):
            _, buf = cv.imencode(".png", img)
            return base64.b64encode(buf).decode("utf-8")

        results.append({
            "filename": f.filename,
            "original": to_base64(img),
            "gradient_mag": to_base64(cv.normalize(gradient_mag, None, 0, 255, cv.NORM_MINMAX)),
            "combined": to_base64(strong_edges + weak_edges),
            "final": to_base64(final_edges)
        })

    return jsonify({"processed_images": results})

# ==============================================================
# MODULE 3 — MANUAL CORNER DETECTION
# ==============================================================
@app.route("/manual_corners")
def manual_corners():
    return render_template("manual_corners.html")

@app.route("/process_manual_corners", methods=["POST"])
def process_manual_corners():
    import base64, math

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    results = []

    for f in files:
        filepath = os.path.join("uploads", f.filename)
        f.save(filepath)

        img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv.GaussianBlur(img, (3, 3), 0)
        H, W = img.shape

        # --- Step 2: Compute gradients manually ---
        Ix = np.zeros_like(img, dtype=np.float32)
        Iy = np.zeros_like(img, dtype=np.float32)
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                gx = (int(img[i, j + 1]) - int(img[i, j - 1])) / 2.0
                gy = (int(img[i + 1, j]) - int(img[i - 1, j])) / 2.0
                Ix[i, j] = gx
                Iy[i, j] = gy

        # --- Step 3: Compute products of derivatives ---
        Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy

        # --- Step 4: Corner response ---
        window_size = 3
        offset = window_size // 2
        R = np.zeros_like(img, dtype=np.float32)
        k = 0.04

        for i in range(offset, H - offset):
            for j in range(offset, W - offset):
                Sxx = np.sum(Ixx[i-offset:i+offset+1, j-offset:j+offset+1])
                Syy = np.sum(Iyy[i-offset:i+offset+1, j-offset:j+offset+1])
                Sxy = np.sum(Ixy[i-offset:i+offset+1, j-offset:j+offset+1])

                det = (Sxx * Syy) - (Sxy ** 2)
                trace = Sxx + Syy
                R[i, j] = det - k * (trace ** 2)

        # --- Step 5: Threshold and mark corners ---
        R_max = np.max(R)
        corner_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        for i in range(H):
            for j in range(W):
                if R[i, j] > 0.01 * R_max:
                    # (0, 0, 255) is RED in BGR
                    cv.circle(corner_img, (j, i), 2, (0, 0, 255), -1)

        # ✅ Convert BGR → RGB to ensure red shows correctly in browser
        corner_img_rgb = cv.cvtColor(corner_img, cv.COLOR_BGR2RGB)

        # --- Encode to base64 for web ---
        def to_base64(im):
            _, buf = cv.imencode(".png", im)
            return base64.b64encode(buf).decode("utf-8")

        results.append({
            "filename": f.filename,
            "original": to_base64(img),
            "corners": to_base64(corner_img_rgb)
        })

    return jsonify({"processed_images": results})

# ==============================================================
# MODULE 3 — INTERACTIVE BOUNDARY DETECTION
# ==============================================================
@app.route("/boundary_detection")
def boundary_detection():
    return render_template("boundary_detection.html")


@app.route("/process_boundary_detection_web", methods=["POST"])
def process_boundary_detection_web():
    import base64

    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    # ROI values (already in same scale as uploaded canvas)
    x = int(float(request.form.get("x", 0)))
    y = int(float(request.form.get("y", 0)))
    w = int(float(request.form.get("w", 0)))
    h = int(float(request.form.get("h", 0)))

    # Read uploaded (scaled) image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Failed to read image"}), 400

    H, W = img.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)

    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return jsonify({"error": "Invalid ROI region"}), 400

    # --- Preprocessing ---
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # --- Edge Detection ---
    edges = cv.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # --- Contour Detection ---
    contours, _ = cv.findContours(edges_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boundary = roi.copy()
    if contours:
        cv.drawContours(boundary, contours, -1, (0, 0, 255), 2)

    # --- Encode to base64 ---
    def to_b64(im):
        _, buf = cv.imencode(".png", im)
        return base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "selected": to_b64(cv.cvtColor(roi, cv.COLOR_BGR2RGB)),
        "edges": to_b64(edges_closed),
        "boundary": to_b64(cv.cvtColor(boundary, cv.COLOR_BGR2RGB))
    })

# ==============================================================
# MODULE 3 — ARUCO-BASED OBJECT SEGMENTATION
# ==============================================================

@app.route("/aruco_segmentation")
def aruco_segmentation():
    return render_template("aruco_segmentation.html")


@app.route("/process_aruco_segmentation", methods=["POST"])
def process_aruco_segmentation():
    """
    Non-rectangular object segmentation using ArUco markers on the boundary.
    Markers should be placed along the object boundary at various positions.
    Uses marker CENTERS to define boundary points, then creates smooth polygon.
    """
    import base64
    import cv2
    import numpy as np
    from scipy.interpolate import splprep, splev  # type: ignore

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    # Use DICT_4X4_50 for ArUco markers (can detect markers with IDs 0-49)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    results = []
    total_markers_detected = 0
    total_images = len(files)

    def to_b64(img):
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode("utf-8")

    def get_marker_centers(corners):
        """Extract center point of each detected ArUco marker."""
        centers = []
        for corner in corners:
            # Each corner is shape (1, 4, 2) - 4 corners of the marker
            pts = corner.reshape(4, 2)
            center = np.mean(pts, axis=0)
            centers.append(center)
        return np.array(centers)

    def order_points_clockwise(pts):
        """Order points in clockwise order around centroid for proper polygon."""
        if len(pts) < 3:
            return pts
        centroid = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        return pts[np.argsort(angles)]

    def create_smooth_boundary(pts, smoothness=0):
        """Create smooth boundary curve through marker centers using spline interpolation."""
        if len(pts) < 4:
            return pts
        
        try:
            # Close the curve by appending first point
            pts_closed = np.vstack([pts, pts[0]])
            
            # Fit spline
            tck, u = splprep([pts_closed[:, 0], pts_closed[:, 1]], s=smoothness, per=True)
            
            # Evaluate spline at more points for smooth curve
            u_new = np.linspace(0, 1, 100)
            x_new, y_new = splev(u_new, tck)
            
            return np.column_stack([x_new, y_new])
        except:
            return pts

    def create_mask_from_polygon(img_shape, polygon):
        """Create binary mask from polygon points."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if polygon is not None and len(polygon) >= 3:
            poly = polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [poly], 255)
        return mask

    def draw_detection_overlay(img, corners, ids, centers, boundary):
        """Draw detected markers, centers, and boundary on image."""
        out = img.copy()
        
        # Draw detected ArUco markers
        if corners is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
        
        # Draw marker centers as circles
        for i, center in enumerate(centers):
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(out, (cx, cy), 8, (255, 0, 0), -1)  # Blue filled circle
            cv2.circle(out, (cx, cy), 10, (255, 255, 255), 2)  # White outline
            # Label with marker ID if available
            if ids is not None and i < len(ids):
                cv2.putText(out, str(ids[i][0]), (cx + 12, cy - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw boundary polygon (green)
        if boundary is not None and len(boundary) >= 3:
            poly = boundary.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [poly], True, (0, 255, 0), 3)
        
        # Add info text
        cv2.putText(out, f"Markers: {len(centers)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return out

    def calculate_iou(mask1, mask2):
        """Calculate Intersection over Union between two masks."""
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        if union == 0:
            return 0.0
        return intersection / union

    for f in files:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(gray)
        num_markers = len(corners) if corners else 0
        total_markers_detected += num_markers

        if ids is None or len(corners) < 3:
            # Need at least 3 markers to form a polygon
            results.append({
                "filename": f.filename,
                "num_markers": num_markers,
                "detection_overlay_b64": to_b64(img),
                "boundary_overlay_b64": to_b64(img),
                "aruco_mask_b64": to_b64(np.zeros((h, w), dtype=np.uint8)),
                "sam2_overlay_b64": to_b64(img),
                "sam2_mask_b64": to_b64(np.zeros((h, w), dtype=np.uint8)),
                "sam2_available": False,
                "iou_score": 0.0,
                "status": "Need at least 3 markers"
            })
            continue

        # Get marker centers (boundary points)
        centers = get_marker_centers(corners)
        
        # Order points clockwise for proper polygon
        ordered_centers = order_points_clockwise(centers)
        
        # Create smooth boundary through marker centers
        smooth_boundary = create_smooth_boundary(ordered_centers, smoothness=0)
        
        # Create detection overlay (shows markers + centers)
        detection_overlay = draw_detection_overlay(img, corners, ids, centers, None)
        
        # Create boundary overlay (shows final boundary)
        boundary_overlay = draw_detection_overlay(img, corners, ids, centers, smooth_boundary)
        
        # Create ArUco-based mask
        aruco_mask = create_mask_from_polygon(img.shape, smooth_boundary)
        
        # --- SAM2 Segmentation for Comparison ---
        sam2_mask = np.zeros((h, w), dtype=np.uint8)
        sam2_overlay = img.copy()
        sam2_used = False
        iou_score = 0.0
        
        if SAM2_AVAILABLE and sam2_predictor is not None:
            try:
                # Get bounding box from marker centers
                x1 = max(0, int(np.min(centers[:, 0])) - 10)
                y1 = max(0, int(np.min(centers[:, 1])) - 10)
                x2 = min(w, int(np.max(centers[:, 0])) + 10)
                y2 = min(h, int(np.max(centers[:, 1])) + 10)
                
                # Run SAM2 with bounding box prompt
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                sam2_predictor.set_image(rgb_img)
                
                box = np.array([x1, y1, x2, y2])
                masks, scores, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=False
                )
                
                sam2_mask = (masks[0] * 255).astype(np.uint8)
                
                # Create SAM2 overlay
                sam2_overlay = img.copy()
                overlay_color = np.zeros_like(img)
                overlay_color[masks[0]] = [0, 0, 255]  # Red
                sam2_overlay = cv2.addWeighted(img, 0.6, overlay_color, 0.4, 0)
                
                # Draw bounding box
                cv2.rectangle(sam2_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(sam2_overlay, "SAM2 Segmentation", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                sam2_used = True
                
                # Calculate IoU between ArUco and SAM2 masks
                iou_score = calculate_iou(aruco_mask, sam2_mask)
                
            except Exception as e:
                print(f"SAM2 error for {f.filename}: {e}")

        results.append({
            "filename": f.filename,
            "num_markers": num_markers,
            "detection_overlay_b64": to_b64(detection_overlay),
            "boundary_overlay_b64": to_b64(boundary_overlay),
            "aruco_mask_b64": to_b64(aruco_mask),
            "sam2_overlay_b64": to_b64(sam2_overlay),
            "sam2_mask_b64": to_b64(sam2_mask),
            "sam2_available": sam2_used,
            "iou_score": round(iou_score * 100, 1),
            "status": "Success"
        })

    # Calculate summary statistics
    avg_markers = total_markers_detected / total_images if total_images > 0 else 0
    successful = sum(1 for r in results if r["status"] == "Success")
    
    return render_template(
        "aruco_segmentation.html", 
        results=results, 
        sam2_enabled=SAM2_AVAILABLE,
        total_images=total_images,
        avg_markers=round(avg_markers, 1),
        successful_count=successful
    )



# ==============================================================
# MODULE 4 — IMAGE STITCHING (PANORAMA CREATION)
# ==============================================================

@app.route("/image_stitching")
def image_stitching():
    """Render the image stitching upload page."""
    return render_template("image_stitching.html")


@app.route("/process_image_stitching", methods=["POST"])
def process_image_stitching():
    """
    Perform panorama stitching on images named 'stitch-*' or uploaded,
    save result as 'stitched_result.jpg', and display side-by-side comparison
    with 'mobile_panorama.jpg' if available.
    """
    import base64
    import cv2
    import numpy as np
    import os

    # --- Step 1: Handle upload (optional) ---
    files = request.files.getlist("images")
    if files:
        for f in files:
            f.save(os.path.join("uploads", f.filename))

    # --- Step 2: Collect all stitch-* images ---
    image_files = sorted([
        f for f in os.listdir("uploads")
        if f.lower().startswith("stitch-") and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(image_files) < 2:
        return jsonify({"error": "Please ensure at least 2 images named 'stitch-1', 'stitch-2', ... are in uploads."}), 400

    imgs = [cv2.imread(os.path.join("uploads", f)) for f in image_files if cv2.imread(os.path.join("uploads", f)) is not None]

    # --- Step 3: Perform Stitching ---
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(imgs)
    if status != cv2.Stitcher_OK:
        return jsonify({"error": f"Stitching failed (code {status})"}), 400

    # --- Step 4: Save stitched output ---
    stitched_path = os.path.join("uploads", "stitched_result.jpg")
    cv2.imwrite(stitched_path, stitched)

    # --- Step 5: Try to load mobile panorama if it exists ---
    mobile_path = os.path.join("uploads", "mobile_panorama.jpeg")
    stitched_img = cv2.imread(stitched_path)
    mobile_img = cv2.imread(mobile_path) if os.path.exists(mobile_path) else None

    # Resize for consistent height
    height = 400
    stitched_resized = cv2.resize(stitched_img, (int(stitched_img.shape[1] * height / stitched_img.shape[0]), height))
    if mobile_img is not None:
        mobile_resized = cv2.resize(mobile_img, (int(mobile_img.shape[1] * height / mobile_img.shape[0]), height))
        combined = np.hstack((stitched_resized, mobile_resized))
        _, buf_combined = cv2.imencode(".jpg", combined)
        combined_b64 = base64.b64encode(buf_combined).decode("utf-8")
    else:
        combined_b64 = None

    # --- Step 6: Encode individual images to Base64 ---
    _, buf_stitched = cv2.imencode(".jpg", stitched_resized)
    stitched_b64 = base64.b64encode(buf_stitched).decode("utf-8")

    stitched_info = {
        "stitched": stitched_b64,
        "combined": combined_b64,
        "files_used": image_files,
        "mobile_found": mobile_img is not None
    }
    return jsonify(stitched_info)

# ==============================================================
# MODULE 4B — SIFT FEATURE EXTRACTION + RANSAC OPTIMIZATION
# ==============================================================

@app.route("/sift_feature_matching")
def sift_feature_matching():
    """Render upload page for SIFT feature extraction experiment."""
    return render_template("sift_feature_matching.html")


@app.route("/process_sift_feature_matching", methods=["POST"])
def process_sift_feature_matching():
    """
    From-scratch SIFT (DoG + Descriptor) + RANSAC Homography estimation.
    Compares results with OpenCV’s SIFT.
    Returns two base64-encoded visualizations.
    """
    import base64, os, cv2, numpy as np

    file1 = request.files.get("image1")
    file2 = request.files.get("image2")
    if not file1 or not file2:
        return jsonify({"error": "Please upload two images."}), 400

    p1 = os.path.join("uploads", file1.filename)
    p2 = os.path.join("uploads", file2.filename)
    file1.save(p1)
    file2.save(p2)

    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)
    if img1 is None or img2 is None:
        return jsonify({"error": "Failed to read uploaded images."}), 400

    # ------------- Simple From-Scratch SIFT Core -------------
    def to_gray_f32(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return g.astype(np.float32) / 255.0

    def build_gaussian_pyramid(img, num_octaves=4, scales=3, sigma0=1.6):
        k = 2 ** (1.0 / scales)
        pyr = []
        base = img
        for _ in range(num_octaves):
            sigmas = [sigma0]
            for s in range(1, scales + 3):
                sigmas.append(sigmas[-1] * k)
            octave = [cv2.GaussianBlur(base, (0, 0), sigmas[0])]
            for s in range(1, scales + 3):
                sigma = np.sqrt(max(sigmas[s] ** 2 - sigmas[s - 1] ** 2, 1e-8))
                octave.append(cv2.GaussianBlur(octave[-1], (0, 0), sigma))
            pyr.append(octave)
            base = cv2.resize(octave[scales], (base.shape[1] // 2, base.shape[0] // 2))
        return pyr

    def build_dog_pyramid(gauss_pyr):
        return [[octave[i + 1] - octave[i] for i in range(len(octave) - 1)] for octave in gauss_pyr]

    def detect_keypoints(dog_pyr, contrast_thresh=0.03):
        kps = []
        for o, octave in enumerate(dog_pyr):
            for s in range(1, len(octave) - 1):
                img = octave[s]
                h, w = img.shape
                for y in range(1, h - 1):
                    for x in range(1, w - 1):
                        patch = np.stack([octave[s - 1][y - 1:y + 2, x - 1:x + 2],
                                          octave[s][y - 1:y + 2, x - 1:x + 2],
                                          octave[s + 1][y - 1:y + 2, x - 1:x + 2]])
                        v = patch[1, 1, 1]
                        if (v == patch.max() or v == patch.min()) and abs(v) > contrast_thresh:
                            kps.append((o, s, y, x))
        return kps

    def sift_from_scratch(img_bgr):
        g = to_gray_f32(img_bgr)
        g_pyr = build_gaussian_pyramid(g)
        dog_pyr = build_dog_pyramid(g_pyr)
        kps = detect_keypoints(dog_pyr)
        # Compute descriptors using OpenCV’s SIFT engine for convenience
        sift = cv2.SIFT_create()
        keypoints = [cv2.KeyPoint(float(x) * (2 ** o), float(y) * (2 ** o), 3.0)
                     for (o, s, y, x) in kps]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, desc = sift.compute(gray, keypoints)
        return keypoints, desc

    # --- From-Scratch SIFT + RANSAC ---
    kp1, d1 = sift_from_scratch(img1)
    kp2, d2 = sift_from_scratch(img2)
    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        return jsonify({"error": "Insufficient keypoints detected."}), 400

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return jsonify({"error": "Not enough good matches for RANSAC."}), 400

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    H_ours, mask_ours = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    vis_ours = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                               matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=mask_ours.ravel().tolist(),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # --- OpenCV SIFT for comparison ---
    sift_ref = cv2.SIFT_create()
    kp1_ref, d1_ref = sift_ref.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2_ref, d2_ref = sift_ref.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    knn = bf.knnMatch(d1_ref, d2_ref, k=2)
    good_ref = [m for m, n in knn if m.distance < 0.75 * n.distance]
    pts1r = np.float32([kp1_ref[m.queryIdx].pt for m in good_ref])
    pts2r = np.float32([kp2_ref[m.trainIdx].pt for m in good_ref])
    H_ref, mask_ref = cv2.findHomography(pts1r, pts2r, cv2.RANSAC, 5.0)

    vis_ref = cv2.drawMatches(img1, kp1_ref, img2, kp2_ref, good_ref, None,
                              matchColor=(255, 255, 0),
                              singlePointColor=(0, 0, 255),
                              matchesMask=mask_ref.ravel().tolist(),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    def to_b64(im):
        _, buf = cv2.imencode(".jpg", im)
        return base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "ours": to_b64(vis_ours),
        "opencv": to_b64(vis_ref),
        "ours_inliers": int(mask_ours.sum()) if mask_ours is not None else 0,
        "opencv_inliers": int(mask_ref.sum()) if mask_ref is not None else 0,
        "matches_ours": len(good),
        "matches_opencv": len(good_ref)
    })

@app.route("/module5")
def module5():
    return render_template("module5.html")

@app.route("/module5/marker")
def module5_marker():
    return render_template("module5_marker.html")

@app.route("/module5/markerless")
def module5_markerless():
    return render_template("module5_markerless.html")

@app.route("/module5/sam2")
def module5_sam2():
    return render_template("module5_sam2.html")

roi_coords = None   # global variable
tracker = None      # initialize here also
last_roi = None
# ============= MediaPipe Pose + Hand Tracking Globals =============
if MEDIAPIPE_AVAILABLE:
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_pose = None
    mp_hands = None
    mp_drawing = None

pose_model = None
hands_model = None

POSE_CSV_PATH = os.path.join("uploads", "pose_hand_data.csv")


@app.route("/start_markerless", methods=["POST"])
def start_markerless():
    global roi_coords, tracker, last_roi, current_bbox
    roi_coords = None
    tracker = None
    last_roi = None
    current_bbox = None
    print("Markerless tracking reset.")
    return jsonify({"status": "ok"})

from flask import request, jsonify

# ... (other imports)

roi_coords = None   # global variable
tracker = None      
last_roi = None
current_bbox = None

@app.route("/set_roi", methods=["POST"])
def set_roi():
    global roi_coords
    # This route receives the full JSON object from JavaScript
    data = request.get_json() 
    roi_coords = data 
    print(f"ROI received and stored: {roi_coords}")
    return jsonify({"status": "ok"})



def gen_frames(process_type=None):
    import cv2
    import cv2.aruco as aruco
    import numpy as np

    # -------------------------------
    # Video Source (Webcam or SAM2 Video)
    # -------------------------------
    if process_type == "sam2":
        cap = None   # prevent webcam use
    else:
        cap = cv2.VideoCapture(0)  # webcam

    # MAIN LOOP
    while True:

        # Normal webcam processing (marker + markerless)
        if process_type != "sam2":
            success, frame = cap.read()
            if not success:
                break

        # ----------------------------------
        # (i) Marker-based Tracking (ArUco)
        # ----------------------------------
        if process_type == "marker":
            if "aruco_dict" not in globals():
                global aruco_dict, parameters
                aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
                parameters = aruco.DetectorParameters()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)

        # ----------------------------------
        # (ii) Markerless Tracking
        # ----------------------------------
        elif process_type == "markerless":
            global roi_coords, tracker, last_roi, current_bbox
            
            # Get actual OpenCV frame size
            frame_h, frame_w = frame.shape[:2]
            
            # -----------------------------------------------------------
            # CASE 1: ROI NOT SELECTED YET
            # -----------------------------------------------------------
            if roi_coords is None:
                cv2.putText(frame, "Draw ROI on the video...", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            else:
                # -----------------------------------------------------------
                # CASE 2: NEW ROI SELECTED (user selected again)
                # -----------------------------------------------------------
                if last_roi != roi_coords:
                    tracker = None   # reset tracker so it can be reinitialized
                    last_roi = roi_coords.copy()

                    # 1. Browser canvas size
                    canvas_w = roi_coords["canvas_w"]
                    canvas_h = roi_coords["canvas_h"]

                    # 2. Scaling from canvas → actual frame
                    scale_x = frame_w / canvas_w
                    scale_y = frame_h / canvas_h

                    # 3. Scale ROI coordinates
                    ocv_x = int(roi_coords["x"] * scale_x)
                    ocv_y = int(roi_coords["y"] * scale_y)
                    ocv_w = int(roi_coords["w"] * scale_x)
                    ocv_h = int(roi_coords["h"] * scale_y)
                    
                    # Ensure valid bounding box
                    ocv_x = max(0, min(ocv_x, frame_w - 1))
                    ocv_y = max(0, min(ocv_y, frame_h - 1))
                    ocv_w = max(10, min(ocv_w, frame_w - ocv_x))
                    ocv_h = max(10, min(ocv_h, frame_h - ocv_y))

                    # Store bounding box globally
                    current_bbox = (ocv_x, ocv_y, ocv_w, ocv_h)

                # -----------------------------------------------------------
                # Initialize tracker ONCE
                # -----------------------------------------------------------
                if tracker is None and current_bbox is not None:
                    # Use CSRT for better accuracy, or KCF for speed
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, current_bbox)
                    print(f"Tracker initialized with bbox: {current_bbox}")

                # -----------------------------------------------------------
                # Update tracker every frame
                # -----------------------------------------------------------
                if tracker is not None:
                    success, updated_bbox = tracker.update(frame)

                    if success:
                        x, y, w, h = [int(v) for v in updated_bbox]
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        # Draw center point
                        cx, cy = x + w // 2, y + h // 2
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(frame, "Tracking", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Tracking lost - Select new ROI", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ----------------------------------
        # (iii) SAM2 Segmentation
        # ----------------------------------
        elif process_type == "sam2":
            import numpy as np
            # Initialize SAM2 video + masks only once
            if "sam2_cap" not in globals():
                print("Loading SAM2 video + masks...")

                global sam2_cap, sam2_masks, sam2_index
                sam2_cap = cv2.VideoCapture("static/sam2/car_video.mp4")
                sam2_masks = np.load("static/sam2/masks.npz")["masks"]
                sam2_index = 0

            # Read next video frame
            success, frame = sam2_cap.read()
            if not success:
                # Restart video when finished
                sam2_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                sam2_index = 0
                continue

            # Select the correct mask for this frame
            if sam2_index >= len(sam2_masks):
                sam2_index = 0

            mask = sam2_masks[sam2_index]
            sam2_index += 1

            # Safety: Resize mask to video frame size
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # Create overlay: mask region becomes red
            overlay = frame.copy()
            overlay[mask == 1] = (0, 0, 255)   # Red for segmentation

            # Blend original + overlay
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # ----------------------------------
        # (iv) Pose + Hand Tracking (MediaPipe)
        # ----------------------------------
        elif process_type == "pose_hand":
            global pose_model, hands_model

            if not MEDIAPIPE_AVAILABLE:
                cv2.putText(frame, "MediaPipe not available (requires Python 3.8-3.12)", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Initialize models once
                if pose_model is None or hands_model is None:
                    pose_model = mp_pose.Pose(
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    hands_model = mp_hands.Hands(
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )

                # Convert to RGB as MediaPipe expects
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run pose + hand tracking
                results_pose = pose_model.process(rgb)
                results_hands = hands_model.process(rgb)

                # Draw pose landmarks
                if results_pose.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results_pose.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                # Draw hand landmarks
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                        )

                # ---------- Save landmarks to CSV ----------
                rows = []
                timestamp = datetime.now().isoformat()

                # Pose landmarks (33 points)
                if results_pose.pose_landmarks:
                    for idx, lm in enumerate(results_pose.pose_landmarks.landmark):
                        # type, hand_index(-1 for pose), landmark_index, timestamp, x, y, z, visibility
                        rows.append([
                            "pose", -1, idx, timestamp,
                            lm.x, lm.y, lm.z, lm.visibility
                        ])

                # Hand landmarks (21 points per hand)
                if results_hands.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                        for idx, lm in enumerate(hand_landmarks.landmark):
                            rows.append([
                                "hand", hand_idx, idx, timestamp,
                                lm.x, lm.y, lm.z, ""  # no visibility provided
                            ])

                if rows:
                    file_exists = os.path.exists(POSE_CSV_PATH)
                    with open(POSE_CSV_PATH, "a", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow([
                                "type",          # 'pose' or 'hand'
                                "hand_index",    # -1 for pose, 0/1 for left/right hand
                                "landmark_index",
                                "timestamp_iso",
                                "x_norm",
                                "y_norm",
                                "z_norm",
                                "visibility"
                            ])
                        writer.writerows(rows)

        # ----------------------------------
        # Stream Frame to Browser
        # ----------------------------------
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ==============================================================
# MODULE 7 — STEREO-BASED OBJECT SIZE ESTIMATION
# ==============================================================

@app.route("/stereo_size_estimation")
def stereo_size_estimation():
    return render_template("stereo_size_estimation.html")


@app.route("/process_stereo_size")
def process_stereo_size():
    global stereo_left, stereo_right, stereo_calib, stereo_roi

    if stereo_left is None:
        return "Please upload images first."

    Q = stereo_calib["Q"]
    if Q is None:
        return "Q matrix missing from calibration file"


    # Compute disparity
    stereo = cv.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7)
    disp = stereo.compute(
        cv.cvtColor(stereo_left, cv.COLOR_BGR2GRAY),
        cv.cvtColor(stereo_right, cv.COLOR_BGR2GRAY)
    ).astype(np.float32) / 16.0

    points_3D = cv.reprojectImageTo3D(disp, Q)

    # Extract ROI
    x = int(stereo_roi["x"])
    y = int(stereo_roi["y"])
    w = int(stereo_roi["w"])
    h = int(stereo_roi["h"])

    roi_pts = points_3D[y:y+h, x:x+w].reshape(-1, 3)

    # Remove invalid depths
    roi_pts = roi_pts[np.isfinite(roi_pts).all(axis=1)]

    # Compute bounding box in 3D
    min_xyz = roi_pts.min(axis=0)
    max_xyz = roi_pts.max(axis=0)
    dims = max_xyz - min_xyz
    width, height, depth = dims

    # disparity viz
    disp_vis = cv.normalize(disp, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    _, buf = cv.imencode(".png", disp_vis)
    disparity_b64 = base64.b64encode(buf).decode("utf-8")

    return render_template(
        "stereo_size_estimation.html",
        dims=f"Width={width:.3f}m, Height={height:.3f}m, Depth={depth:.3f}m",
        disparity_b64=disparity_b64
    )

import base64

stereo_left = None
stereo_right = None
stereo_calib = None
stereo_roi = None

@app.route("/stereo_preview", methods=["POST"])
def stereo_preview():
    global stereo_left, stereo_right, stereo_calib

    stereo_left = cv.imdecode(
        np.frombuffer(request.files["left"].read(), np.uint8), cv.IMREAD_COLOR)
    stereo_right = cv.imdecode(
        np.frombuffer(request.files["right"].read(), np.uint8), cv.IMREAD_COLOR)

    stereo_calib = np.load(request.files["calib"])

    # Convert left image to base64 to display in canvas
    _, buf = cv.imencode(".png", stereo_left)
    left_b64 = base64.b64encode(buf).decode("utf-8")

    return render_template("stereo_size_estimation.html", left_b64=left_b64)

@app.route("/set_stereo_roi", methods=["POST"])
def set_stereo_roi():
    global stereo_roi
    stereo_roi = request.get_json()
    print("Stereo ROI:", stereo_roi)
    return {"status": "ok"}


# ==============================================================
# MODULE 7 — STEREO CALIBRATION (.NPZ GENERATOR)
# ==============================================================

@app.route("/stereo_calibration")
def stereo_calibration():
    return render_template("stereo_calibration.html")


@app.route("/process_stereo_calibration", methods=["POST"])
def process_stereo_calibration():
    import cv2
    import numpy as np
    import os

    left_files = request.files.getlist("left_images")
    right_files = request.files.getlist("right_images")

    if len(left_files) != len(right_files):
        return render_template("stereo_calibration.html", message="Left and right file counts must match!")

    cols = int(request.form.get("chess_cols", 9))
    rows = int(request.form.get("chess_rows", 6))
    pattern_size = (cols, rows)

    # 3D object points for chessboard
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints = []  # 3D real world points
    imgpoints_left = []  # 2D left camera points
    imgpoints_right = []  # 2D right camera points

    for lf, rf in zip(left_files, right_files):

        left = cv2.imdecode(np.frombuffer(lf.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        right = cv2.imdecode(np.frombuffer(rf.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        ret_l, corners_l = cv2.findChessboardCorners(left, pattern_size)
        ret_r, corners_r = cv2.findChessboardCorners(right, pattern_size)

        if ret_l and ret_r:
            objpoints.append(objp)

            corners_l2 = cv2.cornerSubPix(left, corners_l, (11, 11), (-1, -1),
                                          criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            corners_r2 = cv2.cornerSubPix(right, corners_r, (11, 11), (-1, -1),
                                          criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            imgpoints_left.append(corners_l2)
            imgpoints_right.append(corners_r2)

    # Calibrate each camera
    ret_l, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, left.shape[::-1], None, None)
    ret_r, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, right.shape[::-1], None, None)

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K1, D1, K2, D2,
        left.shape[::-1],
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags
    )

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2,
                                               left.shape[::-1],
                                               R, T)

    # Save .npz file
    np.savez("stereo_params.npz",
             K1=K1, D1=D1,
             K2=K2, D2=D2,
             R=R, T=T,
             P1=P1, P2=P2,
             Q=Q)

    return render_template("stereo_calibration.html",
                           message="Calibration successful! stereo_params.npz generated.")

# ==============================================================
# MODULE 8 — REAL-TIME POSE & HAND TRACKING (MEDIAPIPE)
# ==============================================================

@app.route("/pose_hand_tracking")
def pose_hand_tracking():
    return render_template("pose_hand_tracking.html")


@app.route("/video_feed_pose_hand")
def video_feed_pose_hand():
    if IS_CLOUD:
        return jsonify({"error": "Webcam features require local deployment. Run 'python app.py' on your computer."}), 503
    return Response(
        gen_frames(process_type="pose_hand"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/download_pose_data")
def download_pose_data():
    """Download the pose_hand_data.csv file."""
    from flask import send_file
    if os.path.exists(POSE_CSV_PATH):
        return send_file(
            POSE_CSV_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="pose_hand_data.csv"
        )
    else:
        return jsonify({"error": "No data recorded yet. Start tracking first."}), 404


@app.route("/clear_pose_data", methods=["POST"])
def clear_pose_data():
    """Clear/reset the pose_hand_data.csv file before new tracking session."""
    if os.path.exists(POSE_CSV_PATH):
        os.remove(POSE_CSV_PATH)
    return jsonify({"status": "ok", "message": "Previous tracking data cleared."})


@app.route("/video_feed_marker")
def video_feed_marker():
    if IS_CLOUD:
        return jsonify({"error": "Webcam features require local deployment. Run 'python app.py' on your computer."}), 503
    return Response(gen_frames(process_type="marker"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_markerless")
def video_feed_markerless():
    if IS_CLOUD:
        return jsonify({"error": "Webcam features require local deployment. Run 'python app.py' on your computer."}), 503
    return Response(gen_frames(process_type="markerless"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_sam2")
def video_feed_sam2():
    return Response(gen_frames(process_type="sam2"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/generate_sam2_masks")
def generate_sam2_masks():
    try:
        print("\nRunning SAM2 mask generator script...")
        subprocess.run([sys.executable, "generate_sam2_masks_bbox.py"], check=True)
        return jsonify({"status": "ok"})
    except Exception as e:
        print("Error:", e)
        return jsonify({"status": "error", "message": str(e)})


@app.route("/sam2_first_frame")
def sam2_first_frame():
    """Get the first frame of the SAM2 video as base64."""
    video_path = os.path.join("static", "sam2", "car_video.mp4")
    
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found. Please place car_video.mp4 in static/sam2/"}), 404
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video file"}), 500
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": "Could not read video frame"}), 500
    
    # Encode frame as base64
    _, buffer = cv.imencode(".jpg", frame)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")
    
    return jsonify({
        "frame": frame_b64,
        "width": frame.shape[1],
        "height": frame.shape[0]
    })


# Global flag to cancel mask generation
mask_generation_cancelled = False

@app.route("/cancel_mask_generation", methods=["POST"])
def cancel_mask_generation():
    """Cancel ongoing mask generation."""
    global mask_generation_cancelled
    mask_generation_cancelled = True
    print("Mask generation cancelled by user.")
    return jsonify({"status": "ok", "message": "Cancellation requested"})


@app.route("/generate_sam2_masks_web", methods=["POST"])
def generate_sam2_masks_web():
    """Generate SAM2 masks using ROI selected from web interface."""
    global mask_generation_cancelled
    mask_generation_cancelled = False  # Reset flag
    import numpy as np
    
    roi = request.get_json()
    if not roi or "x" not in roi:
        return jsonify({"error": "No ROI provided"}), 400
    
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    print(f"Generating masks with ROI: x={x}, y={y}, w={w}, h={h}")
    
    video_path = os.path.join("static", "sam2", "car_video.mp4")
    output_dir = os.path.join("static", "sam2")
    output_npz = os.path.join(output_dir, "masks.npz")
    
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video"}), 500
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    tracked_masks = []
    
    # Initialize tracker
    ret, first_frame = cap.read()
    if not ret:
        return jsonify({"error": "Could not read video"}), 500
    
    tracker = cv.TrackerCSRT_create()
    tracker.init(first_frame, (x, y, w, h))
    current_bbox = [x, y, w, h]
    
    # Choose segmentation method
    using_sam2 = SAM2_AVAILABLE and sam2_predictor is not None
    method_name = "SAM2" if using_sam2 else "GrabCut"
    print(f"Using {method_name} for segmentation...")
    
    # Process first frame
    if using_sam2:
        mask = generate_sam2_mask(first_frame, current_bbox)
    else:
        mask = generate_grabcut_mask(first_frame, current_bbox)
    tracked_masks.append(mask)
    
    frame_idx = 1
    while True:
        # Check if cancelled
        if mask_generation_cancelled:
            cap.release()
            print("Mask generation cancelled.")
            return jsonify({"status": "cancelled", "message": "Generation cancelled by user"})
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Update tracker
        success, tracked_bbox = tracker.update(frame)
        if success:
            current_bbox = [int(v) for v in tracked_bbox]
        
        # Generate mask using SAM2 or GrabCut
        if using_sam2:
            mask = generate_sam2_mask(frame, current_bbox)
        else:
            mask = generate_grabcut_mask(frame, current_bbox)
        tracked_masks.append(mask)
        
        if frame_idx % 20 == 0:
            print(f"  [{method_name}] Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    
    # Save masks
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(output_npz, masks=np.array(tracked_masks, dtype=np.uint8))
    
    print(f"Saved {len(tracked_masks)} masks to {output_npz}")
    return jsonify({
        "status": "ok",
        "message": f"Generated masks for {len(tracked_masks)} frames"
    })


def generate_sam2_mask(frame, bbox):
    """Generate segmentation mask using SAM2."""
    global sam2_predictor
    
    x, y, w, h = [int(v) for v in bbox]
    
    # Convert BGR to RGB for SAM2
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # Set image in predictor
    sam2_predictor.set_image(rgb_frame)
    
    # Create box prompt [x1, y1, x2, y2]
    box = np.array([x, y, x + w, y + h])
    
    # Get prediction
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False
    )
    
    # Return the mask (convert bool to uint8)
    return masks[0].astype(np.uint8)


def generate_grabcut_mask(frame, bbox):
    """Generate segmentation mask using GrabCut (fallback)."""
    import numpy as np
    
    x, y, w, h = [int(v) for v in bbox]
    
    # Ensure bbox is within frame bounds
    frame_h, frame_w = frame.shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(10, min(w, frame_w - x))
    h = max(10, min(h, frame_h - y))
    
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        rect = (x, y, w, h)
        cv.grabCut(frame, mask, rect, bgd_model, fgd_model, 3, cv.GC_INIT_WITH_RECT)
        result_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    except:
        # Fallback: simple rectangular mask
        result_mask = np.zeros(frame.shape[:2], np.uint8)
        result_mask[y:y+h, x:x+w] = 1
    
    return result_mask


if __name__ == "__main__":
    app.run(debug=True)

