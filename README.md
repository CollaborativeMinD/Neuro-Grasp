# Neuro-Grasp: PCA-Based Robotic Grasp Detection

**Neuro-Grasp** is a modular computer vision system designed to calculate 6-DOF grasp poses for unknown objects without relying on large machine learning datasets. It utilizes **Principal Component Analysis (PCA)** to geometrically determine an object's primary axis (Orientation) and secondary axis (Grasp Approach Vector) in real-time.

## 🎯 Project Goal
To build a robust, low-latency perception system that can identify the optimal "Pick Angle" for a robotic end-effector purely from 2D contour data. This approach is deterministic, explainable, and computationally cheaper than neural networks for rigid industrial parts.

## 📐 The Math (Why it works)
Instead of guessing, we calculate the object's **Eigenvectors** from the covariance matrix of its contour points.

* **First Eigenvector (Major Axis):** Defines the object's length/orientation.
* **Second Eigenvector (Minor Axis):** Defines the object's width (The Approach Vector).
* **Center of Mass:** The geometric centroid for the suction/gripper target.

## 🛠️ Technical Workflow
1.  **Auto-Configuration:** The script automatically verifies and installs missing dependencies upon launch.
2.  **Resilient Acquisition:** A self-healing driver manages the camera feed, automatically resetting the connection if signal loss is detected.
3.  **Preprocessing:** Converts RGB → Grayscale → Binary Threshold.
4.  **PCA Solver:** Calculates the mean and eigenvectors of the contour cloud to derive the grasp pose.
5.  **Kinematics Bridge:** Translates pixel coordinates into Robot Frame (mm) telemetry.

## 🚀 Usage
The script is self-contained. Run it directly from the terminal:

```bash
python Neuro_Grasp.py
