# Neo_Grasp.py
# A Modular Vision System for Robotic Grasping using PCA

# LOGIC BLOCK I: IMPORTS & DEPENDENCIES
import sys
import subprocess
import math
import random
import os

# The "Auto-Installer" - Keeps your environment portable
def install_and_import(package, import_name=None):
    if import_name is None: import_name = package
    try:
        __import__(import_name)
    except ImportError:
        print(f"[INFO] {package} not found. Installing automatically...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[INFO] {package} installed. Continuing...")

# Summoning the heavy lifters
install_and_import("numpy")
install_and_import("opencv-python", "cv2")
install_and_import("seaborn")
install_and_import("matplotlib")

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("SYSTEM GREEN: Libraries loaded and ready.")

# %%
# LOGIC BLOCK II: CONFIGURATION
# The "Rules of Physics" for your simulation
IMG_SIZE = 600
OUTPUT_FILE = "grasp_result.png"

# Visualization colors (BGR Format for OpenCV)
COLOR_BACKGROUND = (0, 0, 0)
COLOR_OBJECT = (255, 255, 255)
COLOR_CENTER = (0, 255, 255) # Yellow
COLOR_MAJOR_AXIS = (0, 0, 255) # Red
COLOR_MINOR_AXIS = (255, 0, 0) # Blue

# %%
# LOGIC BLOCK III: (TOOLKIT) DATA ACQUISITION & DRIVERS
import time

def generate_synthetic_data(simulate_failure=False):
    """
    HARDWARE SIMULATOR:
    Generates a black image with a random white rectangle.
    Raises ConnectionError if 'simulate_failure' is True.
    """
    if simulate_failure:
        raise ConnectionError("CRITICAL: Camera Feed Lost - Signal Interrupted")

    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # Random dimensions for the "part"
    w, h = random.randint(100, 300), random.randint(50, 100)
    angle = random.randint(0, 360)
    center = (random.randint(150, 450), random.randint(150, 450))
    
    rect = ((center[0], center[1]), (w, h), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    cv2.drawContours(img, [box], 0, COLOR_OBJECT, -1)
    
    # Only print "Generated" if successful (reduces log noise)
    # print(f"DEBUG: Generated part at {center}") 
    return img

def acquire_camera_feed(chaos_mode=False):
    """
    THE DRIVER (Self-Healing):
    Attempts to acquire an image. 
    If hardware fails, it automatically pauses, resets, and retries.
    Returns: Image (numpy array) or None (if critical failure).
    """
    try:
        # Attempt 1: Normal Acquisition
        return generate_synthetic_data(simulate_failure=chaos_mode)

    except ConnectionError as e:
        # The Recovery Logic (Encapsulated here, not in the main loop)
        print(f"   ⚠️ [DRIVER WARNING] {e}")
        print("   >>> [DRIVER ACTION] Connection Unstable. Initiating Reset (3s)...")
        time.sleep(3) # The physical pause
        
        try:
            print("   >>> [DRIVER ACTION] Reconnecting...")
            # Retry with failure disabled (Simulating a successful reboot)
            img = generate_synthetic_data(simulate_failure=False) 
            print("   ✅ [DRIVER STATUS] Connection Restored.")
            return img
            
        except ConnectionError:
            print("   ❌ [DRIVER FAILURE] Manual Intervention Required.")
            return None
        
def pixel_to_robot_frame(pixel_center, pixel_angle):
    """
    [PLACEHOLDER] - To be updated.
    Converts pixel coordinates to robot frame coordinates.
    """
    # 1. Define the Scale (e.g., 100 pixels = 10mm)
    PIXEL_TO_MM_RATIO = 0.1 # 1 pixel = 0.1 mm
    # 2. Convert pixel coordinates to mm
    mm_x = pixel_center[0] * PIXEL_TO_MM_RATIO
    mm_y = pixel_center[1] * PIXEL_TO_MM_RATIO
    # 3. Angle Translation (Robot might grip at +90 deg offset)
    robot_wrist_angle = pixel_angle + 90
    # 4. Apply any necessary transformations (e.g., origin shift, rotation)
    'ArithmeticError: Placeholder for future kinematic transformations.'
    '# Steps to implement:'
    # 5. Return the transformed coordinates and angle
    return (mm_x, mm_y, robot_wrist_angle)



# %%
# LOGIC BLOCK IV: GNC ALGORITHMS (PCA)
def calculate_grasp_orientation(pts, img):
    """
    Performs PCA (Principal Component Analysis) to find the primary axis.
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
    cv2.circle(img, cntr, 5, COLOR_CENTER, -1)

    # --- CALCULATE GRASP AXIS ---
    scale_factor = 150 
    
    # Major Axis (Red) - Orientation
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0] * scale_factor,
          cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0] * scale_factor)
    
    # Minor Axis (Blue) - The Approach Vector
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0] * scale_factor,
          cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0] * scale_factor)

    # Draw Axis
    cv2.arrowedLine(img, cntr, (int(p1[0]), int(p1[1])), COLOR_MAJOR_AXIS, 3, tipLength=0.1)
    cv2.arrowedLine(img, cntr, (int(p2[0]), int(p2[1])), COLOR_MINOR_AXIS, 3, tipLength=0.1)
    
    # Calculate Angle in degrees
    angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0]) * 180 / math.pi
    
    return angle, cntr

# %%
# LOGIC BLOCK V: MISSION EXECUTION (FLIGHT DIRECTOR)
def execute_mission_sequence(total_cycles=3):
    print(f"--- Neuro-Grasp: Vision System ({total_cycles} Cycle Test) ---")

    for i in range(1, total_cycles + 1):
        print(f"\n[CYCLE {i}/{total_cycles}] Pinging Sensor Array...")

        # --- PHASE 1: ACQUISITION ---
        print("[1/3] Acquiring Camera Feed...")
        # We pass 'chaos_mode=True' to randomly test the failsafe inside the driver.
        is_sabotage = (random.random() < 0.2)
        execution_img = acquire_camera_feed(chaos_mode=is_sabotage)

        # CRITICAL CHECK: If the driver returns None, it means even the retry failed.
        if execution_img is None:
            print("   ❌ [MISSION ABORT] Sensor Failure. Skipping Cycle.")
            continue 

        # --- PHASE 2: PROCESSING ---
        print("[2/3] Processing Topography...")
        gray = cv2.cvtColor(execution_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        print(f"[3/3] Detected {len(contours)} potential grasp targets.")

        target_found = False 
        for j, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < 1000: continue
            
            # Execute PCA Logic from Block IV
            angle, center = calculate_grasp_orientation(c, execution_img)
            
            label = f"Angle: {int(angle)} deg"
            cv2.putText(execution_img, label, (center[0] - 100, center[1] - 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"   -> TARGET {j}: Center={center} | Grasp Angle={angle:.2f}")
            target_found = True
            
            # Send to Robot Frame Conversion (Placeholder)
            
            # 1. Get the Vision Data (Pixels)
            angle, center = calculate_grasp_orientation(c, execution_img)

            # 2. CONVERT TO ROBOT DATA (Millimeters) <--- The new line
            robot_x, robot_y, robot_grip = pixel_to_robot_frame(center, angle)

            # 3. Print the "Real World" Coordinates
            print(f"   -> VISION: {center} px | ROBOT: ({robot_x:.1f}, {robot_y:.1f}) mm")
        
        # --- PHASE 3: VISUALIZATION ---
        if target_found:
            plt.figure(figsize=(6,6)) 
            plt.imshow(cv2.cvtColor(execution_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Cycle {i}: Target Locked")
            plt.axis('off')
            plt.show()
        else:
            print("   [INFO] No valid targets found in this scan.")

    print("\n--- MISSION COMPLETE. SYSTEM STANDBY. ---")

# Execute the loop
execute_mission_sequence(total_cycles=5)

