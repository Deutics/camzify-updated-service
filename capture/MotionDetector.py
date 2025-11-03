# MotionDetector.py
import cv2
import numpy as np
from collections import deque


class MotionDetection:
    """
    Simple motion detector using frame differencing and contour detection.
    Expects grayscale input frames.
    """

    def __init__(self, min_area: int = 300, history_frames: int = 15, threshold: int = 20):
        """
        Args:
            min_area: Minimum contour area to consider as motion
            history_frames: Number of frames to keep in history for background subtraction
            threshold: Threshold value for binary thresholding (lower = more sensitive)
        """
        self.min_area = min_area
        self.history_frames = history_frames
        self.threshold = threshold
        self.frame_history = deque(maxlen=history_frames)
        self.prev_frame = None
        self.initialized = False
        # Create a proper kernel for morphological operations
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray) -> bool:
        """
        Detect motion in the given grayscale frame.

        Args:
            frame: Grayscale frame (single channel, CV_8UC1)

        Returns:
            True if motion detected, False otherwise
        """
        if frame is None:
            return False

        # Ensure frame is grayscale
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        # Initialize previous frame if needed
        if self.prev_frame is None:
            self.prev_frame = blurred
            self.initialized = False
            return False

        # Skip first few frames to let the detector stabilize
        if not self.initialized:
            self.prev_frame = blurred
            self.initialized = True
            return True  # Assume motion on first real check to trigger ACTIVE mode

        # Compute absolute difference between current and previous frame
        frame_delta = cv2.absdiff(self.prev_frame, blurred)

        # Threshold the delta image
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, self.kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contour is large enough
        motion_detected = False
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
            if area >= self.min_area:
                motion_detected = True
                # Don't break - continue to find max area for debugging

        # Update previous frame
        self.prev_frame = blurred

        # Update history
        self.frame_history.append(blurred)

        return motion_detected

    def reset(self):
        """Reset the motion detector state."""
        self.prev_frame = None
        self.initialized = False
        self.frame_history.clear()