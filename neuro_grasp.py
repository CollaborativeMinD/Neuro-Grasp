"""
Neuro-Grasp: PCA-Based Robotic Grasp Detection System.

Refactored for high-integrity industrial environments.
Implements deterministic PCA for orientation calculation with 
real-time vector visualization.

Architecture:
    - Model: VisionProcessor (PCA Logic), GraspPose (Data Object)
    - Hardware: SyntheticCamera (Driver Layer)
    - Controller: GraspOrchestrator (Mission Logic)

Author: Senior Systems Architect (Refactored)
Original Author: Charles Austin
"""

# pylint: disable=no-member

import sys
import time
import math
import random
import logging
import unittest
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CELL 1: IMPORTS & CONFIGURATION ---

# Configure System Logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("NeuroGrasp")

@dataclass(frozen=True)
class VisionConfig:
    """Immutable parameters for image processing."""
    img_size: int = 600
    min_area: float = 1000.0
    binary_threshold: int = 50
    scale_factor: int = 150
    # BGR Colors
    color_center: Tuple[int, int, int] = (0, 255, 255) # Yellow
    color_major: Tuple[int, int, int] = (0, 0, 255)   # Red
    color_minor: Tuple[int, int, int] = (255, 0, 0)   # Blue

@dataclass(frozen=True)
class RobotConfig:
    """Kinematic transformation constants."""
    pixel_to_mm: float = 0.1
    angle_offset_deg: float = 90.0

@dataclass
class GraspPose:
    """Standardized object for passing grasp telemetry."""
    center_px: Tuple[int, int]
    angle_deg: float
    center_mm: Tuple[float, float]
    robot_angle: float
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

# --- CELL 2: CLASS DEFINITIONS & LOGIC ---

class SyntheticCamera:
    """Simulates a camera driver with self-healing capabilities."""

    def __init__(self, size: int = 600) -> None:
        self.size = size
        self.frame_count = 0

    def get_status(self) -> str:
        """Returns the health status of the driver."""
        return f"ONLINE: {self.frame_count} frames processed"

    def reset_driver(self) -> None:
        """Resets the internal driver state."""
        self.frame_count = 0
        LOGGER.info("Camera Driver Reset.")

    def _generate_frame(self, simulate_failure: bool) -> np.ndarray:
        """Internal method to generate synthetic part geometry."""
        if simulate_failure:
            raise ConnectionError("Driver Fault: Camera Signal Interrupted.")

        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        width, height = random.randint(100, 300), random.randint(50, 100)
        angle = random.randint(0, 360)
        center = (random.randint(150, 450), random.randint(150, 450))

        rect = ((center[0], center[1]), (width, height), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(img, [box], 0, (255, 255, 255), -1)
        return img

    def get_frame(self, chaos_mode: bool = False) -> Optional[np.ndarray]:
        """Acquire frame with PACE recovery logic."""
        try:
            frame = self._generate_frame(simulate_failure=chaos_mode)
            self.frame_count += 1
            return frame
        except ConnectionError as err:
            LOGGER.warning("Hardware glitch detected: %s. Retrying...", err)
            time.sleep(1.0)
            try:
                frame = self._generate_frame(simulate_failure=False)
                self.frame_count += 1
                return frame
            except ConnectionError:
                LOGGER.error("Critical Driver Failure. Manual reset required.")
                return None

class VisionProcessor:
    """Performs geometric analysis and coordinate transformation."""

    def __init__(self, v_cfg: VisionConfig, r_cfg: RobotConfig) -> None:
        self.v_cfg = v_cfg
        self.r_cfg = r_cfg

    def get_config_summary(self) -> Dict[str, Any]:
        """Returns current operational parameters for system audit."""
        return {
            "img_size": self.v_cfg.img_size,
            "min_area": self.v_cfg.min_area,
            "px_to_mm": self.r_cfg.pixel_to_mm
        }

    def calculate_grasp_pose(self, contour: np.ndarray) -> GraspPose:
        """Executes PCA to find orientation and maps to robot coordinates."""
        data_pts = contour.reshape(-1, 2).astype(np.float64)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, np.array([]))

        center_px = (int(mean[0, 0]), int(mean[0, 1]))
        angle_rad = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])
        angle_deg = math.degrees(angle_rad)

        mm_x = center_px[0] * self.r_cfg.pixel_to_mm
        mm_y = center_px[1] * self.r_cfg.pixel_to_mm
        robot_angle = angle_deg + self.r_cfg.angle_offset_deg

        return GraspPose(
            center_px=center_px,
            angle_deg=angle_deg,
            center_mm=(mm_x, mm_y),
            robot_angle=robot_angle,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors
        )

    def draw_telemetry(self, img: np.ndarray, pose: GraspPose) -> None:
        """Overlays PCA vectors (Red/Blue) and data on the frame."""
        cntr = pose.center_px
        cv2.circle(img, cntr, 5, self.v_cfg.color_center, -1)
        scale = 0.02 * self.v_cfg.scale_factor
        p1 = (int(cntr[0] + pose.eigenvectors[0, 0] * pose.eigenvalues[0, 0] * scale),
              int(cntr[1] + pose.eigenvectors[0, 1] * pose.eigenvalues[0, 0] * scale))
        p2 = (int(cntr[0] - pose.eigenvectors[1, 0] * pose.eigenvalues[1, 0] * scale),
              int(cntr[1] - pose.eigenvectors[1, 1] * pose.eigenvalues[1, 0] * scale))
        cv2.arrowedLine(img, cntr, p1, self.v_cfg.color_major, 3, tipLength=0.1)
        cv2.arrowedLine(img, cntr, p2, self.v_cfg.color_minor, 3, tipLength=0.1)
        label = f"Angle: {int(pose.angle_deg)} deg"
        cv2.putText(img, label, (cntr[0] - 100, cntr[1] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# --- CELL 3: UNIT TESTS ---

class TestVisionLogic(unittest.TestCase):
    """Verifies deterministic math and configuration gates."""

    def setUp(self) -> None:
        self.v_cfg = VisionConfig()
        self.r_cfg = RobotConfig()
        self.processor = VisionProcessor(self.v_cfg, self.r_cfg)

    def test_coordinate_transformation(self) -> None:
        """Test that pixel to mm conversion logic remains stable."""
        dummy_contour = np.array([[[100, 100]], [[110, 100]], [[110, 110]]])
        pose = self.processor.calculate_grasp_pose(dummy_contour)
        self.assertAlmostEqual(pose.center_mm[0], 10.6, delta=0.5)

    def test_driver_recovery(self) -> None:
        """Verify the driver handles signal loss gracefully."""
        cam = SyntheticCamera()
        frame = cam.get_frame(chaos_mode=True)
        self.assertIsNotNone(frame, "Driver failed to recover from transient error")

# --- CELL 4: MISSION EXECUTION ---

class GraspOrchestrator:
    """High-level mission controller."""

    def __init__(self) -> None:
        self.v_cfg = VisionConfig()
        self.r_cfg = RobotConfig()
        self.camera = SyntheticCamera(self.v_cfg.img_size)
        self.processor = VisionProcessor(self.v_cfg, self.r_cfg)
        self.mission_active = False

    def get_mission_state(self) -> bool:
        """Returns the current status of the mission."""
        return self.mission_active

    def run_mission(self, cycles: int = 3) -> None:
        """Executes the perception loop with real-time visualization."""
        self.mission_active = True
        print(f"\n--- NEURO-GRASP ACTIVE: {cycles} CYCLE TEST ---")
        for i in range(1, cycles + 1):
            print(f"[CYCLE {i}/{cycles}] Scanning Array...")
            is_unstable = random.random() < 0.2
            frame = self.camera.get_frame(chaos_mode=is_unstable)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(gray, self.v_cfg.binary_threshold, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            target_found = False
            for contour in cnts:
                if cv2.contourArea(contour) < self.v_cfg.min_area:
                    continue
                pose = self.processor.calculate_grasp_pose(contour)
                self.processor.draw_telemetry(frame, pose)
                print(f"  ✅ TARGET DETECTED: Angle={pose.angle_deg:.1f}°")
                print(f"  🤖 ROBOT TELEMETRY: ({pose.center_mm[0]:.1f}, {pose.center_mm[1]:.1f}) mm")
                target_found = True
            if target_found:
                plt.figure(figsize=(6, 6))
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title(f"Cycle {i}: PCA Grasp Vector")
                plt.axis('off')
                plt.show(block=False)
                plt.pause(1)
                plt.close()
        self.mission_active = False
        print("--- MISSION COMPLETE. SYSTEM STANDBY. ---")

if __name__ == "__main__":
    # 1. Pre-flight unit test verification
    LOADER = unittest.TestLoader()
    SUITE = LOADER.loadTestsFromTestCase(TestVisionLogic)
    RUNNER = unittest.TextTestRunner(verbosity=0)
    # 2. Execute tests and launch mission if successful
    RESULT = RUNNER.run(SUITE)
    if RESULT.wasSuccessful():
        ORCH = GraspOrchestrator()
        ORCH.run_mission(cycles=5)
    else:
        sys.exit(1)
