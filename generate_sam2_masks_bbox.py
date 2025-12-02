"""
SAM2 Video Segmentation Mask Generator

This script generates segmentation masks for a video using either:
1. SAM2 (if available and properly installed)
2. Fallback: OpenCV GrabCut + Tracking (if SAM2 is not available)

Usage: python generate_sam2_masks_bbox.py
"""

import cv2
import numpy as np
import os
import sys

# --- 1. CONFIGURATION ---
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "static", "sam2", "car_video.mp4")
output_dir = os.path.join(base_dir, "static", "sam2")
output_npz_path = os.path.join(output_dir, "masks.npz")

print("=" * 60)
print("VIDEO SEGMENTATION MASK GENERATOR")
print("=" * 60)

# --- 2. CHECK VIDEO EXISTS ---
if not os.path.exists(video_path):
    print(f"ERROR: Video not found at: {video_path}")
    print("Please place 'car_video.mp4' in the 'static/sam2/' folder.")
    sys.exit(1)

# --- 3. TRY TO LOAD SAM2 ---
sam2_available = False
sam2_predictor = None

try:
    import torch
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    
    checkpoint_path = os.path.join(base_dir, "sam2_checkpoints", "sam2_hiera_tiny.pt")
    model_cfg = os.path.join(base_dir, "configs", "sam2_hiera_t.yaml")
    
    if os.path.exists(checkpoint_path) and os.path.exists(model_cfg):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SAM2 model on {device.upper()}...")
        
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        sam2_available = True
        print("✅ SAM2 loaded successfully!")
    else:
        print("⚠️ SAM2 checkpoint or config not found.")
        
except ImportError as e:
    print(f"⚠️ SAM2 not installed: {e}")
except Exception as e:
    print(f"⚠️ SAM2 loading failed: {e}")

if not sam2_available:
    print("\n📢 Using fallback method: OpenCV GrabCut + Tracking")
    print("   (Install SAM2 for better results)\n")


# --- 4. OPEN VIDEO AND GET INITIAL BOUNDING BOX ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"ERROR: Could not open video: {video_path}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames @ {fps:.1f} FPS")

ret, first_frame = cap.read()
if not ret:
    print("ERROR: Could not read first frame.")
    sys.exit(1)

# Get bounding box from user
print("\n📌 Draw a bounding box around the object to segment, then press ENTER.")
bbox = cv2.selectROI("Select Object to Segment", first_frame, showCrosshair=True)
cv2.destroyWindow("Select Object to Segment")

x, y, w, h = [int(v) for v in bbox]
if w < 10 or h < 10:
    print("ERROR: Invalid bounding box. Please select a larger area.")
    sys.exit(1)

print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")


# --- 5. SEGMENTATION FUNCTIONS ---

def segment_with_sam2(frame, bbox, predictor):
    """Use SAM2 to segment the object."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    
    x, y, w, h = bbox
    box = np.array([x, y, x + w, y + h])
    
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False
    )
    
    return masks[0].astype(np.uint8)


def segment_with_grabcut(frame, bbox):
    """Fallback: Use OpenCV GrabCut for segmentation."""
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    rect = (bbox[0], bbox[1], bbox[2], bbox[3])  # x, y, w, h
    
    try:
        cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        result_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    except:
        # If GrabCut fails, return a simple rectangular mask
        result_mask = np.zeros(frame.shape[:2], np.uint8)
        result_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1
    
    return result_mask


# --- 6. PROCESS VIDEO ---
print("\n--- Processing Video ---")
tracked_masks = []

# Initialize tracker for bounding box propagation
tracker = cv2.TrackerCSRT_create()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, frame = cap.read()
tracker.init(frame, (x, y, w, h))

frame_idx = 0
current_bbox = [x, y, w, h]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    
    # Update tracker to get new bbox
    if frame_idx > 1:
        success, tracked_bbox = tracker.update(frame)
        if success:
            current_bbox = [int(v) for v in tracked_bbox]
        # If tracking fails, keep the last known bbox
    
    # Segment the frame
    if sam2_available:
        mask = segment_with_sam2(frame, current_bbox, sam2_predictor)
    else:
        mask = segment_with_grabcut(frame, current_bbox)
    
    tracked_masks.append(mask)
    
    # Progress indicator
    if frame_idx % 10 == 0 or frame_idx == total_frames:
        progress = (frame_idx / total_frames) * 100
        print(f"  Processing: {frame_idx}/{total_frames} frames ({progress:.1f}%)")

cap.release()
cv2.destroyAllWindows()


# --- 7. SAVE OUTPUT ---
os.makedirs(output_dir, exist_ok=True)
np.savez_compressed(output_npz_path, masks=np.array(tracked_masks, dtype=np.uint8))

print(f"\n✅ DONE! Saved {len(tracked_masks)} masks to:")
print(f"   {output_npz_path}")
print("\nYou can now use the SAM2 demo in your web app!")
