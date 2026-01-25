import sys
import subprocess
import math
import random
import os

# --- AUTO-INSTALLER (Ensures OpenCV and NumPy exist) ---
def install_and_import(package, import_name=None):
    if import_name is None: import_name = package
    try:
        __import__(import_name)
    except ImportError:
        print(f"[INFO] {package} not found. Installing automatically...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[INFO] {package} installed. Continuing...")

install_and_import("numpy")
install_and_import("opencv-python", "cv2")

import cv2
import numpy as np

# --- CONFIGURATION ---
IMG_SIZE = 600
OUTPUT_FILE = "grasp_result.png"

def generate_synthetic_data():
    """Generates a black image with a random white rectangle (The 'Part')."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # Random dimensions for the "part"
    w, h = random.randint(100, 300), random.randint(50, 100)
    angle = random.randint(0, 360)
    center = (random.randint(150, 450), random.randint(150, 450))
    
    # Create the rotated rectangle
    rect = ((center[0], center[1]), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # Draw filled white rectangle on black background
    cv2.drawContours(img, [box], 0, (255, 255, 255), -1)
    return img

def calculate_grasp_orientation(pts, img):
    """
    Performs PCA (Principal Component Analysis) to find the primary axis of the object.
    This replaces 'black box' AI with geometric certainty.
    """
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    # Visualization: Draw the Center Point
    cv2.circle(img, cntr, 5, (0, 255, 255), -1) # Yellow Dot

    # --- CALCULATE GRASP AXIS ---
    # The Eigenvectors point in the direction of the primary axes
    # p1 = Major Axis (Length), p2 = Minor Axis (Width/Grasp Approach)
    
    # Scale lines for visibility
    scale_factor = 150 
    
    # Major Axis (Red) - Orientation
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0] * scale_factor,
          cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0] * scale_factor)
    
    # Minor Axis (Blue) - The Approach Vector (Perpendicular to length)
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0] * scale_factor,
          cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0] * scale_factor)

    # Draw Axis
    cv2.arrowedLine(img, cntr, (int(p1[0]), int(p1[1])), (0, 0, 255), 3, tipLength=0.1)  # RED = Orientation
    cv2.arrowedLine(img, cntr, (int(p2[0]), int(p2[1])), (255, 0, 0), 3, tipLength=0.1)  # BLUE = Approach
    
    # Calculate Angle in degrees
    angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0]) * 180 / math.pi
    
    return angle, cntr

def main():
    print("--- Neuro-Grasp: Vision System ---")
    print("[1/4] Generating Synthetic Part Data...")
    img = generate_synthetic_data()
    
    print("[2/4] Preprocessing (Greyscale -> Threshold -> Contours)...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    print(f"[3/4] Detected {len(contours)} potential objects.")
    
    for i, c in enumerate(contours):
        # Calculate area to filter out noise
        area = cv2.contourArea(c)
        if area < 1000: continue
        
        # Calculate Grasp Orientation
        angle, center = calculate_grasp_orientation(c, img)
        
        # Label the image
        label = f"Grasp Angle: {int(angle)} deg"
        cv2.putText(img, label, (center[0] - 100, center[1] - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f"   -> Object {i}: Center={center}, Angle={angle:.2f}")

    print(f"[4/4] Saving Visualization to '{OUTPUT_FILE}'...")
    cv2.imwrite(OUTPUT_FILE, img)
    print(f"[SUCCESS] Open '{OUTPUT_FILE}' to see the result.")

    # Try to open the image automatically (Windows)
    try:
        os.startfile(OUTPUT_FILE)
    except:
        pass

if __name__ == "__main__":
    main()