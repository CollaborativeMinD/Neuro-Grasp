# Neuro-Grasp: PCA-Based Robotic Grasp Detection

**Neuro-Grasp** is a computer vision module designed to calculate 6-DOF grasp poses for unknown objects without relying on large machine learning datasets. It utilizes **Principal Component Analysis (PCA)** to geometrically determine an object's primary axis (Orientation) and secondary axis (Grasp Approach Vector) in real-time.

## 🎯 Project Goal
To build a robust, low-latency perception system that can identify the optimal "Pick Angle" for a robotic end-effector purely from 2D contour data. This approach is deterministic and computationally cheaper than neural networks for rigid industrial parts.

## 📐 The Math (Why it works)
Instead of guessing, we calculate the object's **Eigenvectors** from the covariance matrix of its contour points.

* **First Eigenvector (Major Axis):** Defines the object's length/orientation.
* **Second Eigenvector (Minor Axis):** Defines the object's width (The Approach Vector).
* **Center of Mass:** The geometric centroid for the suction/gripper target.

## 🛠️ Technical Workflow
1.  **Synthetic Generation:** The script generates random "industrial parts" (rectangles) with randomized rotation/scale to validate the algorithm.
2.  **Preprocessing:** Converts RGB → Grayscale → Binary Threshold.
3.  **Contour Extraction:** `cv2.findContours` isolates the object boundaries.
4.  **PCA Solver:** Calculates the mean and eigenvectors of the contour cloud.
5.  **Visualization:** Projects the 6-DOF pose onto the 2D image.

## 🚀 Usage
The script is self-contained and will verify dependencies automatically.

```bash
python neuro_grasp.py

```

**Output:**

* The script will generate a file named `grasp_result.png`.
* **Red Arrow:** Orientation (Major Axis)
* **Blue Arrow:** Grasp Approach (Minor Axis)
* **Yellow Dot:** Suction Target (Centroid)

## 📦 Dependencies

* Python 3.10+
* OpenCV (`opencv-python`)
* NumPy

---

**Author:** Charles Austin
*Focus: Computer Vision, Robotics Perception, and AI.*
