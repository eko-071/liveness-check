import math

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

CALIBRATION_FRAMES = 30
EAR_THRESHOLD_RATIO = 0.75
MIN_CLOSED_FRAMES = 2


def dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def compute_ear(landmarks, eye_indices):
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical_1 = dist(p2, p6)
    vertical_2 = dist(p3, p5)
    horizontal = dist(p1, p4)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


class BlinkDetector:
    def __init__(self):
        self.calibration_ears = []
        self.baseline_ear = None
        self.threshold = None
        self.closed_frames = 0
        self.blink_count = 0
        self.blink_detected_this_frame = False

    @property
    def is_calibrated(self):
        return self.baseline_ear is not None

    def update(self, landmarks):
        self.blink_detected_this_frame = False

        left_ear = compute_ear(landmarks, LEFT_EYE)
        right_ear = compute_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        # Calibration phase
        if not self.is_calibrated:
            self.calibration_ears.append(avg_ear)
            if len(self.calibration_ears) >= CALIBRATION_FRAMES:
                self.baseline_ear = sum(self.calibration_ears) / len(self.calibration_ears)
                self.threshold = self.baseline_ear * EAR_THRESHOLD_RATIO
                print(f"\nBlink calibrated - baseline EAR: {self.baseline_ear:.3f}, threshold: {self.threshold:.3f}")
            return avg_ear, False

        # Blink detection
        if avg_ear < self.threshold:
            self.closed_frames += 1
        else:
            if self.closed_frames >= MIN_CLOSED_FRAMES:
                self.blink_count += 1
                self.blink_detected_this_frame = True
            self.closed_frames = 0

        return avg_ear, self.blink_detected_this_frame