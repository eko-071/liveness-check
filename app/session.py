import random
import time
from app.blink import BlinkDetector
from app.head_pose import HeadPoseDetector
from app.scorer import compute_score

CHALLENGE_TIMEOUT = 7
ALL_CHALLENGES = ["BLINK", "TURN_LEFT", "TURN_RIGHT"]

PROMPTS = {
    "CALIBRATE":  "Hold still, calibrating...",
    "BLINK":      "Please blink",
    "TURN_LEFT":  "Turn your head left",
    "TURN_RIGHT": "Turn your head right",
    "DONE":       "",
}


class LivenessSession:
    """
    Processes one frame at a time and tracks challenge state.
    Knows nothing about where frames come from — webcam or WebSocket.
    """
    def __init__(self):
        self.blink = BlinkDetector()
        self.head = HeadPoseDetector()

        self.order = random.sample(ALL_CHALLENGES, len(ALL_CHALLENGES))
        self.index = 0
        self.state = "CALIBRATE"
        self.challenge_start = None
        self.returned_center = False

        self.checks = {
            ch: {
                "passed": False,
                "timed_out": False,
                "elapsed": None,
                "timeout": CHALLENGE_TIMEOUT,
            }
            for ch in ALL_CHALLENGES
        }

        self.result = None
        self.total_frames = 0
        self.detected_frames = 0

    @property
    def current_challenge(self):
        if self.index < len(self.order):
            return self.order[self.index]
        return None

    def _advance(self, now):
        self.index += 1
        self.returned_center = False
        ch = self.current_challenge
        if ch:
            self.state = ch
            self.challenge_start = now
        else:
            self.state = "DONE"

    def _check_passed(self, ch, blinked, direction):
        if ch == "BLINK":
            return blinked
        if ch == "TURN_LEFT":
            return direction == "LEFT"
        if ch == "TURN_RIGHT":
            if direction == "CENTER":
                self.returned_center = True
            return direction == "RIGHT" and self.returned_center
        return False

    def _update_challenge(self, ch, blinked, direction, now):
        if self.challenge_start is None:
            return

        time_left = max(0.0, CHALLENGE_TIMEOUT - (now - self.challenge_start))

        if self._check_passed(ch, blinked, direction):
            elapsed = CHALLENGE_TIMEOUT - time_left
            self.checks[ch]["passed"] = True
            self.checks[ch]["elapsed"] = round(elapsed, 2)
            self._advance(now)

        elif time_left == 0:
            self.checks[ch]["timed_out"] = True
            self._advance(now)

    def _finalise(self):
        consistency = (self.detected_frames / self.total_frames
                       if self.total_frames > 0 else 0.0)
        self.checks["consistency"] = round(consistency, 3)
        self.result = compute_score(self.checks)

    def process(self, landmarks):
        """
        Feed one frame's worth of landmarks into the session.
        Returns a status dict describing what just happened.
        Call this once per frame, whether from a webcam loop or a WebSocket.
        """
        now = time.time()
        self.total_frames += 1

        if landmarks is None:
            return self._status(time_left=0)

        self.detected_frames += 1
        _, blinked = self.blink.update(landmarks)
        _, direction = self.head.update(landmarks)
        both_ready = self.blink.is_calibrated and self.head.is_calibrated
        ch = self.current_challenge

        if self.state == "CALIBRATE" and both_ready:
            self.state = ch
            self.challenge_start = now

        elif self.state == "DONE":
            if self.result is None:
                self._finalise()

        elif ch:
            self._update_challenge(ch, blinked, direction, now)

        time_left = (
            max(0.0, CHALLENGE_TIMEOUT - (now - self.challenge_start))
            if self.challenge_start and self.state not in ("CALIBRATE", "DONE")
            else 0.0
        )

        return self._status(time_left)

    def _status(self, time_left):
        """
        The standard response shape returned after every frame.
        This is what detector.py reads to update the UI,
        and what api.py sends back over the WebSocket.
        """
        return {
            "state":      self.state,
            "prompt":     PROMPTS.get(self.state, ""),
            "order":      self.order,
            "index":      self.index,
            "checks":     self.checks,
            "time_left":  round(time_left, 2),
            "result":     self.result,
        }