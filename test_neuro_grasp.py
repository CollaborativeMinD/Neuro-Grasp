"""
Unit Test Suite for Neuro-Grasp Vision System.
TOTAL TESTS: 12

This module provides comprehensive coverage for:
1. PCA-Based Orientation Math (Eigenvector validation)
2. Hardware Driver Resilience (PACE self-healing logic)
3. Kinematic Coordinate Transformations (Pixel to MM)
4. System Configuration Integrity

Target Pylint Score: 10.0/10 (With no-member disable)
"""

# pylint: disable=no-member, protected-access

import unittest
import math
from unittest.mock import patch
import numpy as np

# Import refactored components from the primary module
from neuro_grasp import (
    VisionProcessor,
    SyntheticCamera,
    VisionConfig,
    RobotConfig,
    GraspPose
)

class TestVisionProcessor(unittest.TestCase):
    """Verifies geometric analysis and PCA math (5 Tests)."""

    def setUp(self) -> None:
        """Initialize configurations for deterministic testing."""
        self.v_cfg = VisionConfig()
        self.r_cfg = RobotConfig(pixel_to_mm=1.0, angle_offset_deg=0.0)
        self.processor = VisionProcessor(self.v_cfg, self.r_cfg)

    def test_pca_orientation_horizontal(self) -> None:
        """Verify that a horizontal rectangle returns ~0 degrees orientation."""
        # Create a horizontal set of points
        contour = np.array([[[100, 200]], [[500, 200]], [[500, 210]], [[100, 210]]])
        pose = self.processor.calculate_grasp_pose(contour)

        # Orientation should be aligned with X-axis
        cos_val = abs(math.cos(math.radians(pose.angle_deg)))
        self.assertAlmostEqual(cos_val, 1.0, places=2)

    def test_pca_orientation_vertical(self) -> None:
        """Verify that a vertical rectangle returns ~90 degrees orientation."""
        contour = np.array([[[200, 100]], [[200, 500]], [[210, 500]], [[210, 100]]])
        pose = self.processor.calculate_grasp_pose(contour)

        # Orientation should be aligned with Y-axis
        sin_val = abs(math.sin(math.radians(pose.angle_deg)))
        self.assertAlmostEqual(sin_val, 1.0, places=2)

    def test_coordinate_mapping(self) -> None:
        """Verify pixel-to-millimeter scaling logic."""
        custom_r_cfg = RobotConfig(pixel_to_mm=0.5, angle_offset_deg=90.0)
        proc = VisionProcessor(self.v_cfg, custom_r_cfg)

        contour = np.array([[[100, 100]], [[200, 100]], [[200, 200]], [[100, 200]]])
        pose = proc.calculate_grasp_pose(contour)

        # Center is (150, 150) px -> (75.0, 75.0) mm
        self.assertEqual(pose.center_mm, (75.0, 75.0))
        self.assertAlmostEqual(pose.robot_angle, 90.0, places=2)

    def test_empty_contour_handling(self) -> None:
        """Ensure the processor raises appropriate errors on invalid data."""
        with self.assertRaises(Exception):
            self.processor.calculate_grasp_pose(np.array([]))

    def test_pose_data_integrity(self) -> None:
        """Verify the GraspPose object contains all required fields."""
        contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]]])
        pose = self.processor.calculate_grasp_pose(contour)
        self.assertIsInstance(pose, GraspPose)
        self.assertTrue(hasattr(pose, 'eigenvalues'))


class TestHardwareDriver(unittest.TestCase):
    """Verifies self-healing PACE logic in the Synthetic Camera (4 Tests)."""

    def setUp(self) -> None:
        self.camera = SyntheticCamera(size=600)

    def test_initial_state(self) -> None:
        """Verify the driver starts with clean telemetry."""
        self.assertIn("ONLINE: 0", self.camera.get_status())

    def test_successful_acquisition(self) -> None:
        """Test normal operation (Primary Path)."""
        frame = self.camera.get_frame(chaos_mode=False)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (600, 600, 3))
        self.assertEqual(self.camera.frame_count, 1)

    @patch('time.sleep', return_value=None)
    def test_driver_recovery_path(self, _mock_sleep) -> None:
        """Test the self-healing recovery (Alternate Path)."""
        # We use patch.object here for robustness
        with patch.object(SyntheticCamera, '_generate_frame') as mock_gen:
            # First call fails, second succeeds
            mock_gen.side_effect = [ConnectionError("Transient"), np.zeros((600, 600, 3))]
            frame = self.camera.get_frame(chaos_mode=True)
            self.assertIsNotNone(frame)
            self.assertEqual(mock_gen.call_count, 2)

    @patch('time.sleep', return_value=None)
    def test_critical_failure_path(self, _mock_sleep) -> None:
        """Test behavior when hardware remains unresponsive (Emergency Path)."""
        # Patch the internal generator to always fail using the class reference directly
        with patch.object(SyntheticCamera, '_generate_frame',
                          side_effect=ConnectionError("Persistent Failure")):
            frame = self.camera.get_frame(chaos_mode=True)
            self.assertIsNone(frame, "Driver should return None on critical persistent failure")


class TestSystemIntegration(unittest.TestCase):
    """Verifies configuration and high-level logic gates (3 Tests)."""

    def test_vision_config_immutability(self) -> None:
        """Ensure system constants cannot be modified at runtime."""
        cfg = VisionConfig()
        with self.assertRaises(Exception):
            cfg.img_size = 800 # type: ignore

    def test_robot_config_defaults(self) -> None:
        """Verify factory default safety parameters."""
        cfg = RobotConfig()
        self.assertEqual(cfg.pixel_to_mm, 0.1)
        self.assertEqual(cfg.angle_offset_deg, 90.0)

    def test_synthetic_frame_generation(self) -> None:
        """Ensure generated frames contain valid pixel data for PCA."""
        cam = SyntheticCamera(size=600)
        # Attempt to generate a frame. 
        # The generator uses random placement; we verify it produces data.
        frame = cam._generate_frame(simulate_failure=False)
        
        # If the random generator happened to place the object off-screen (rare at 600px),
        # we try one more time to ensure the test isn't flaky.
        if not np.any(frame > 0):
            frame = cam._generate_frame(simulate_failure=False)

        self.assertTrue(np.any(frame > 0), "Generated frame was empty.")

if __name__ == "__main__":
    unittest.main()
