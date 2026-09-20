"""Exercise repetition counter built on MediaPipe Pose Landmarker.

Detects 33 body landmarks, measures the joint angle that defines a given
exercise, and runs a two-state machine over that angle to count repetitions.
"""

import os
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe Pose landmark indices, named so the exercise table reads clearly.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Skeleton edges drawn in both the 2D overlay and the 3D viewer.
SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

VISIBILITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class ExerciseSpec:
    """Defines one exercise as a joint angle plus the two poses that bound it.

    Every supported exercise moves a joint from an *extended* (large angle) pose
    into a *flexed* (small angle) pose and back. One repetition is counted each
    time the athlete returns to the extended pose after having reached flexion,
    which is why the movement must pass through both thresholds to register.
    """

    joint: str              # human-readable joint name, shown on the overlay
    left: tuple             # the three landmark indices forming the left angle
    right: tuple            # the three landmark indices forming the right angle
    flexed_below: float     # at or below this angle the joint counts as flexed
    extended_above: float   # at or above this angle the joint counts as extended
    symmetric: bool         # True when both sides move together (report one total)


EXERCISES = {
    "Bicep Curl": ExerciseSpec(
        joint="Elbow",
        left=(L_SHOULDER, L_ELBOW, L_WRIST),
        right=(R_SHOULDER, R_ELBOW, R_WRIST),
        flexed_below=50, extended_above=150, symmetric=False,
    ),
    "Squat": ExerciseSpec(
        joint="Knee",
        left=(L_HIP, L_KNEE, L_ANKLE),
        right=(R_HIP, R_KNEE, R_ANKLE),
        flexed_below=100, extended_above=160, symmetric=True,
    ),
    "Push-up": ExerciseSpec(
        joint="Elbow",
        left=(L_SHOULDER, L_ELBOW, L_WRIST),
        right=(R_SHOULDER, R_ELBOW, R_WRIST),
        flexed_below=90, extended_above=150, symmetric=True,
    ),
    "Lunge": ExerciseSpec(
        joint="Knee",
        left=(L_HIP, L_KNEE, L_ANKLE),
        right=(R_HIP, R_KNEE, R_ANKLE),
        flexed_below=100, extended_above=160, symmetric=False,
    ),
    "Jumping Jack": ExerciseSpec(
        # Armpit angle: arms at the sides is flexed, arms overhead is extended.
        joint="Shoulder",
        left=(L_HIP, L_SHOULDER, L_WRIST),
        right=(R_HIP, R_SHOULDER, R_WRIST),
        flexed_below=45, extended_above=120, symmetric=True,
    ),
}

EXERCISE_NAMES = list(EXERCISES)


def open_video_writer(path, fps, size):
    """Open a VideoWriter, preferring H.264 so browsers can play the result.

    OpenCV's default mp4v stream is not playable in Chrome or Safari, which
    makes processed video look broken inside the Gradio player even though the
    file itself is fine. avc1 is tried first and mp4v kept as a fallback for
    builds without an H.264 encoder.
    """
    fps = fps if fps and fps > 0 else 30.0
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open a video writer for {path}")


def calculate_angle(a, b, c):
    """Return the angle ABC in degrees, where b is the vertex joint."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def detect_view(landmarks):
    """Guess whether the camera sees the athlete from the front or the side.

    Seen side-on the two shoulders project almost onto each other, so the
    normalised horizontal gap between them collapses.
    """
    try:
        gap = abs(landmarks[L_SHOULDER].x - landmarks[R_SHOULDER].x)
    except (IndexError, AttributeError):
        return "Unknown"
    return "Front" if gap > 0.12 else "Side"


class RepCounter:
    """Tracks angle, movement phase, and repetition count for each side."""

    def __init__(self, exercise="Bicep Curl", smoothing_factor=0.3):
        if exercise not in EXERCISES:
            raise ValueError(
                f"Unknown exercise {exercise!r}. Choose one of: {', '.join(EXERCISE_NAMES)}"
            )
        self.exercise = exercise
        self.spec = EXERCISES[exercise]
        self.smoothing_factor = smoothing_factor
        self.reset()

    def reset(self):
        self.counter = {"left": 0, "right": 0}
        self.stage = {"left": None, "right": None}
        self.smoothed = {"left": None, "right": None}

    @property
    def total_reps(self):
        """Reps to report: one shared count when both sides move together."""
        if self.spec.symmetric:
            return max(self.counter["left"], self.counter["right"])
        return self.counter["left"] + self.counter["right"]

    def smooth_angle(self, new_angle, side):
        """Exponential moving average, to stop landmark jitter faking reps."""
        previous = self.smoothed[side]
        if previous is None:
            self.smoothed[side] = new_angle
        else:
            f = self.smoothing_factor
            self.smoothed[side] = f * new_angle + (1 - f) * previous
        return self.smoothed[side]

    def update_state(self, angle, side):
        """Advance the state machine; count a rep on the return to extension.

        The joint must actually cross *both* thresholds for a rep to register.
        Angles in the band between them leave the phase untouched, so a partial
        movement that never reaches full flexion is correctly ignored.
        """
        if angle >= self.spec.extended_above:
            if self.stage[side] == "flexed":
                self.counter[side] += 1
            self.stage[side] = "extended"
        elif angle <= self.spec.flexed_below:
            self.stage[side] = "flexed"

    def _measure_side(self, landmarks, indices, side):
        """Return the smoothed angle for one side, or None if it is obscured.

        Skipping low-confidence joints means an occluded limb pauses counting
        rather than producing a burst of phantom repetitions.
        """
        try:
            joints = [landmarks[i] for i in indices]
        except IndexError:
            return None
        if any(j.visibility < VISIBILITY_THRESHOLD for j in joints):
            return None

        angle = calculate_angle(*[(j.x, j.y) for j in joints])
        angle = self.smooth_angle(angle, side)
        self.update_state(angle, side)
        return angle

    def process_landmarks(self, landmarks):
        """Measure both sides for the current exercise. Returns (left, right)."""
        return (
            self._measure_side(landmarks, self.spec.left, "left"),
            self._measure_side(landmarks, self.spec.right, "right"),
        )

    def draw_ui(self, image, left_angle, right_angle, detected_view, fps=None):
        """Draw the stats panel over the frame."""
        panel_h = 150
        panel_w = min(430, image.shape[1])

        # Darken the region behind the text so it stays readable on any footage.
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (24, 24, 24), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, dst=image)
        cv2.rectangle(image, (0, 0), (panel_w, panel_h), (245, 117, 16), 2)

        white, amber, grey = (255, 255, 255), (66, 200, 245), (190, 190, 190)

        def angle_text(angle):
            return f"{int(angle)}deg" if angle is not None else "--"

        cv2.putText(image, self.exercise.upper(), (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, amber, 2, cv2.LINE_AA)

        subtitle = f"{self.spec.joint} angle  |  View: {detected_view}"
        if fps is not None:
            subtitle += f"  |  {fps:.0f} FPS"
        cv2.putText(image, subtitle, (14, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, grey, 1, cv2.LINE_AA)

        if self.spec.symmetric:
            cv2.putText(image, f"REPS  {self.total_reps}", (14, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, white, 2, cv2.LINE_AA)
            cv2.putText(image, f"L {angle_text(left_angle)}    R {angle_text(right_angle)}",
                        (14, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.55, grey, 1, cv2.LINE_AA)
        else:
            for label, column, reps, angle in (
                ("LEFT", 14, self.counter["left"], left_angle),
                ("RIGHT", 225, self.counter["right"], right_angle),
            ):
                cv2.putText(image, label, (column, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, grey, 1, cv2.LINE_AA)
                cv2.putText(image, str(reps), (column, 118),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, white, 2, cv2.LINE_AA)
                cv2.putText(image, angle_text(angle), (column, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, grey, 1, cv2.LINE_AA)
        return image


def render_3d_skeleton(world_landmarks):
    """Render the pose world landmarks as a 3D plot, returned as a BGR image."""
    fig = plt.figure(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor('black')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.view_init(elev=15, azim=45)

    # World landmarks are metres relative to the hips, so +/-1m frames the body.
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    tracked = {i for edge in SKELETON for i in edge}
    pts = {}
    for i, lm in enumerate(world_landmarks):
        if i in tracked:
            # MediaPipe world axes are x=right, y=down, z=depth; matplotlib wants
            # z pointing up, so depth becomes the plot's y and y is negated.
            pts[i] = (lm.x, lm.z, -lm.y)

    if pts:
        xs, ys, zs = zip(*pts.values())
        ax.scatter(xs, ys, zs, c='m', marker='o')
    for i, j in SKELETON:
        if i in pts and j in pts:
            ax.plot(*zip(pts[i], pts[j]), c='orange')

    ax.set_axis_off()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return img


class PoseProcessor:
    """Runs pose detection and rep counting over a stream of frames."""

    def __init__(self, exercise="Bicep Curl", show_3d=False):
        self.counter = RepCounter(exercise=exercise)
        self.show_3d = show_3d
        self.last_img_3d = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'pose_landmarker_full.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. "
                "Download pose_landmarker_full.task from the MediaPipe model zoo."
            )

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp_vision.PoseLandmarker.create_from_options(options)

    def close(self):
        """Release the underlying MediaPipe landmarker."""
        if self.pose is not None:
            self.pose.close()
            self.pose = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def draw_custom_landmarks(self, image, landmarks):
        h, w = image.shape[:2]
        pts = {}
        for i, lm in enumerate(landmarks):
            if lm.visibility > VISIBILITY_THRESHOLD:
                pts[i] = (int(lm.x * w), int(lm.y * h))
                cv2.circle(image, pts[i], 4, (245, 66, 230), -1)

        for i, j in SKELETON:
            if i in pts and j in pts:
                cv2.line(image, pts[i], pts[j], (245, 117, 66), 2)
        return image

    def process_frame(self, frame, fps=None):
        """Detect, count, and annotate a single BGR frame."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.pose.detect(mp_image)

        image_bgr = frame.copy()
        left_angle = right_angle = None
        detected_view = "Unknown"
        img_3d = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            detected_view = detect_view(landmarks)
            image_bgr = self.draw_custom_landmarks(image_bgr, landmarks)
            left_angle, right_angle = self.counter.process_landmarks(landmarks)

            if self.show_3d and results.pose_world_landmarks:
                img_3d = render_3d_skeleton(results.pose_world_landmarks[0])

        image_bgr = self.counter.draw_ui(image_bgr, left_angle, right_angle, detected_view, fps)

        if self.show_3d:
            image_bgr = self._append_3d_panel(image_bgr, img_3d)
        return image_bgr

    def _append_3d_panel(self, image_bgr, img_3d):
        """Stack the 3D viewer beside the frame at a rock-steady width.

        Frames where the pose is momentarily lost reuse the previous panel so
        the output resolution never changes mid-video, which would otherwise
        corrupt the encoded stream.
        """
        h_main = image_bgr.shape[0]
        if img_3d is not None:
            scale = h_main / img_3d.shape[0]
            self.last_img_3d = cv2.resize(img_3d, (int(img_3d.shape[1] * scale), h_main))
        elif self.last_img_3d is None:
            self.last_img_3d = np.zeros((h_main, h_main, 3), dtype=np.uint8)
        return np.hstack((image_bgr, self.last_img_3d))


def process_video_file(input_path, output_path, exercise="Bicep Curl", show_3d=False,
                       progress=None):
    """Annotate a video file with the skeleton and rep counter.

    `progress` is an optional callable taking (fraction, description), which
    lets the Gradio UI show a live progress bar during long jobs.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width == 0 or height == 0:
            raise ValueError(f"Video reports an empty frame size: {input_path}")

        # The 3D panel is rendered square, so it adds `height` pixels of width.
        out_width = width + height if show_3d else width
        writer = open_video_writer(str(output_path), fps, (out_width, height))

        processor = PoseProcessor(exercise, show_3d=show_3d)
        try:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                output = processor.process_frame(frame)
                if output.shape[1] != out_width or output.shape[0] != height:
                    output = cv2.resize(output, (out_width, height))
                writer.write(output)

                frame_index += 1
                if progress is not None and total > 0:
                    progress(
                        min(frame_index / total, 1.0),
                        f"Tracking pose: frame {frame_index}/{total}",
                    )

            reps = processor.counter.total_reps
        finally:
            processor.close()
            writer.release()
    finally:
        cap.release()

    return str(output_path), reps
