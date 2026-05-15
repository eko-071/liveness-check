CALIBRATION_FRAMES = 30
YAW_THRESHOLD = 0.4 # how far from baseline to count as a turn

# Landmark indices
NOSE_TIP = 4
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33


def compute_yaw_ratio(landmarks):
    nose = landmarks[NOSE_TIP]
    left_eye = landmarks[LEFT_EYE_OUTER]
    right_eye = landmarks[RIGHT_EYE_OUTER]

    eye_width = right_eye.x - left_eye.x
    if abs(eye_width) < 1e-6:
        return None

    ratio = (nose.x - left_eye.x) / eye_width
    return ratio


class HeadPoseDetector:
    def __init__(self):
        self.calibration_ratios = []
        self.baseline_ratio = None

    @property
    def is_calibrated(self):
        return self.baseline_ratio is not None

    def update(self, landmarks):
        ratio = compute_yaw_ratio(landmarks)
        if ratio is None:
            return ratio, "CENTER"

        # Calibration phase
        if not self.is_calibrated:
            self.calibration_ratios.append(ratio)
            if len(self.calibration_ratios) >= CALIBRATION_FRAMES:
                self.baseline_ratio = sum(self.calibration_ratios) / len(self.calibration_ratios)
                print(f"\nHead pose calibrated — baseline ratio: {self.baseline_ratio:.3f}")
            return ratio, "CALIBRATING"

        delta = ratio - self.baseline_ratio

        if delta < -YAW_THRESHOLD:
            direction = "LEFT"
        elif delta > YAW_THRESHOLD:
            direction = "RIGHT"
        else:
            direction = "CENTER"

        return ratio, direction