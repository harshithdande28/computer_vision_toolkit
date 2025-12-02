import cv2
import numpy as np
import math

# --------------------------
# CONFIGURATION
# --------------------------
DICT_TYPE = cv2.aruco.DICT_4X4_50     # Dictionary to match your Flask code
NUM_MARKERS = 20                      # How many markers you want
MARKER_SIZE = 300                     # Pixels per marker (300 px → ~3–5 cm when printed)
PADDING = 40                          # Space between markers
GRID_COLS = 5                         # Number of markers per row
OUTPUT_FILE = "aruco_marker_sheet.png"  # Final sheet file

# --------------------------
# GENERATE MARKERS
# --------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
marker_images = []

for marker_id in range(NUM_MARKERS):
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE)
    marker_images.append(marker)

# --------------------------
# BUILD GRID SHEET
# --------------------------
rows = math.ceil(NUM_MARKERS / GRID_COLS)
cols = GRID_COLS

sheet_width = cols * MARKER_SIZE + (cols + 1) * PADDING
sheet_height = rows * MARKER_SIZE + (rows + 1) * PADDING

sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255  # White background

index = 0
for r in range(rows):
    for c in range(cols):
        if index >= NUM_MARKERS:
            break
        
        x = c * MARKER_SIZE + (c + 1) * PADDING
        y = r * MARKER_SIZE + (r + 1) * PADDING

        sheet[y:y+MARKER_SIZE, x:x+MARKER_SIZE] = marker_images[index]
        index += 1

# --------------------------
# SAVE RESULT
# --------------------------
cv2.imwrite(OUTPUT_FILE, sheet)
print(f"Marker sheet saved as {OUTPUT_FILE}")
