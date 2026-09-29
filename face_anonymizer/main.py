"""Occlusion-aware face anonymizer.

Detects faces with MediaPipe BlazeFace, then optionally refines each detection
into a pixel-accurate mask with a face-parsing segmentation model so that
objects in front of the face stay sharp while the face itself is blurred.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import torch
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FACE_PARSING_MODEL = "jonathandinu/face-parsing"

# Face-parsing classes that carry identity, and so must be blurred. Hair (13),
# clothing and accessories are deliberately excluded: they do not identify a
# person the way facial features do, and leaving them sharp looks far more
# natural than a blurred halo around the head.
IDENTITY_CLASSES = [
    1,   # skin
    2,   # nose
    3,   # eyeglasses
    4, 5,    # left eye, right eye
    6, 7,    # left brow, right brow
    10,      # mouth
    11, 12,  # upper lip, lower lip
]

# Fraction by which each detected box is grown before blurring. BlazeFace
# returns a tight crop that can clip the chin and hairline, which would leave
# an identifiable sliver unblurred at the edges.
BOX_PADDING = 0.3

# Resolution the face crop is resized to before segmentation. Cost scales
# sharply with this: on Apple MPS a 512x512 pass takes ~133ms (7.5 FPS) while
# 256x256 takes ~46ms (21.5 FPS). The default caps quality-first file
# processing; live video lowers it so the window stays responsive.
MAX_PARSE_SIZE = 512
LIVE_PARSE_SIZE = 320
# Below ~192px the parsing model stops recognising a face at all: a 128px pass
# over a webcam face labelled 0% of it as skin, so perfect mode blurred
# nothing. 256px found the face reliably.
MIN_PARSE_SIZE = 256


def open_video_writer(path, fps, size):
    """Open a VideoWriter, preferring H.264 so browsers can play the result.

    OpenCV's default mp4v stream is not playable in Chrome or Safari, which
    makes processed video look broken inside the Gradio player even though the
    file itself is fine. avc1 is tried first and mp4v kept as a fallback for
    builds without an H.264 encoder.
    """
    fps = fps if fps and fps > 0 else 30.0
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open a video writer for {path}")


def select_device():
    """Pick the best available torch device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class FaceAnonymizer:
    """Blurs faces in a frame, either by bounding box or by semantic mask."""

    def __init__(self, mode='perfect', blur_intensity=51, min_detection_confidence=0.5,
                 parse_size=MAX_PARSE_SIZE):
        self.mode = mode
        self.blur_intensity = blur_intensity
        self.parse_size = parse_size

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'blaze_face_short_range.tflite')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face detection model not found at {model_path}. "
                "Download blaze_face_short_range.tflite from the MediaPipe model zoo."
            )

        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=min_detection_confidence,
        )
        self.face_detection = mp_vision.FaceDetector.create_from_options(options)

        self.device = None
        self.processor = None
        self.model = None
        if self.mode == 'perfect':
            self._load_parsing_model()

    @property
    def blur_intensity(self):
        return self._blur_intensity

    @blur_intensity.setter
    def blur_intensity(self, value):
        """Keep the blur kernel odd, as GaussianBlur requires."""
        value = int(value)
        self._blur_intensity = value if value % 2 != 0 else value + 1
        self.blur_kernel = (self._blur_intensity, self._blur_intensity)

    def _load_parsing_model(self):
        self.device = select_device()
        print(f"Loading face parsing model on {self.device} (downloads ~150MB on first run)...")
        self.processor = SegformerImageProcessor.from_pretrained(FACE_PARSING_MODEL)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            FACE_PARSING_MODEL
        ).to(self.device).eval()

    def set_mode(self, mode):
        """Switch modes, loading the parsing model on first use of 'perfect'."""
        if mode == 'perfect' and self.model is None:
            self._load_parsing_model()
        self.mode = mode

    def detect_boxes(self, frame):
        """Return padded, clamped (x1, y1, x2, y2) boxes for every face found."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.face_detection.detect(mp_image)
        if not results.detections:
            return []

        h, w = frame.shape[:2]
        boxes = []
        for detection in results.detections:
            bbox = detection.bounding_box
            x = int(bbox.origin_x - bbox.width * BOX_PADDING / 2)
            y = int(bbox.origin_y - bbox.height * BOX_PADDING / 2)
            width = int(bbox.width * (1 + BOX_PADDING))
            height = int(bbox.height * (1 + BOX_PADDING))

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + width), min(h, y + height)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _segmentation_size(self, rois):
        """Choose the resolution to segment at, based on the actual crop size.

        A detected face is often only ~200px across, so forcing it up to
        512x512 spends roughly three times the compute to invent detail that
        was never in the pixels. Sizing to the crop (rounded to a multiple of
        32 for the model's patch grid) is dramatically faster on large crops.
        Crops smaller than MIN_PARSE_SIZE are still upsampled to it, because
        the model cannot find a face in a very small input.
        """
        largest = max(max(roi.shape[0], roi.shape[1]) for roi in rois)
        target = min(self.parse_size, max(MIN_PARSE_SIZE, largest))
        return int(round(target / 32) * 32)

    def _parse_faces(self, rois):
        """Segment several face crops in one forward pass.

        Batching matters in group shots: the segmentation model dominates the
        per-frame cost, so running it once for N faces rather than N times
        keeps a crowded frame close to the speed of a single-face frame.
        """
        size = self._segmentation_size(rois)
        pil_images = [Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) for roi in rois]
        inputs = self.processor(
            images=pil_images, return_tensors="pt",
            size={"height": size, "width": size},
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        masks = []
        for i, roi in enumerate(rois):
            resized = torch.nn.functional.interpolate(
                logits[i : i + 1],
                size=roi.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            preds = resized.argmax(dim=1).squeeze(0).cpu().numpy()
            mask = np.isin(preds, IDENTITY_CLASSES).astype(np.float32)
            # Feather the mask so the blur fades in rather than cutting a
            # hard-edged hole, which reads as an obvious artefact.
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            masks.append(mask[:, :, np.newaxis])
        return masks

    def process_frame(self, frame):
        """Return a copy of the frame with every detected face anonymized."""
        boxes = self.detect_boxes(frame)
        if not boxes:
            # No faces: hand back the frame untouched rather than paying for
            # a pointless copy or risking a crash on an empty detection list.
            return frame

        output_frame = frame.copy()
        rois = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]
        blurred = [cv2.GaussianBlur(roi, self.blur_kernel, 0) for roi in rois]

        if self.mode == 'perfect' and self.model is not None:
            masks = self._parse_faces(rois)
            for (x1, y1, x2, y2), roi, blur, mask in zip(boxes, rois, blurred, masks):
                output_frame[y1:y2, x1:x2] = (blur * mask + roi * (1.0 - mask)).astype(np.uint8)
        else:
            for (x1, y1, x2, y2), blur in zip(boxes, blurred):
                output_frame[y1:y2, x1:x2] = blur

        return output_frame

    def process_image_file(self, input_path, output_path):
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")
        cv2.imwrite(str(output_path), self.process_frame(img))
        return str(output_path)

    def process_video_file(self, input_path, output_path, progress=None):
        """Anonymize every frame of a video.

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

            writer = open_video_writer(output_path, fps, (width, height))
            try:
                from tqdm import tqdm

                bar = tqdm(total=total if total > 0 else None, desc="Anonymizing")
                frame_index = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(self.process_frame(frame))
                    frame_index += 1
                    bar.update(1)
                    if progress is not None and total > 0:
                        progress(
                            min(frame_index / total, 1.0),
                            f"Anonymizing: frame {frame_index}/{total}",
                        )
                bar.close()
            finally:
                writer.release()
        finally:
            cap.release()

        return str(output_path)

    def close(self):
        if self.face_detection is not None:
            self.face_detection.close()
            self.face_detection = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def run_webcam(anonymizer):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.", file=sys.stderr)
        return

    print("Press 'q' to quit. Press 'm' to toggle between 'basic' and 'perfect' mode.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            output = anonymizer.process_frame(frame)
            fps = 1.0 / max(time.time() - start_time, 1e-6)

            cv2.putText(output, f"FPS: {fps:.1f} | Mode: {anonymizer.mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Anonymizer", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m'):
                anonymizer.set_mode('basic' if anonymizer.mode == 'perfect' else 'perfect')
                print(f"Switched to {anonymizer.mode} mode")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Occlusion-aware face anonymizer")
    parser.add_argument('--input', default='webcam',
                        help="Path to an input image or video, or 'webcam'")
    parser.add_argument('--output', default=None,
                        help="Output path (default: anonymized_<input name> beside the input)")
    parser.add_argument('--mode', choices=['basic', 'perfect'], default='perfect',
                        help="'basic' blurs the whole box; 'perfect' respects occlusions")
    parser.add_argument('--blur', type=int, default=51, help="Blur strength (odd number)")
    args = parser.parse_args()

    anonymizer = FaceAnonymizer(mode=args.mode, blur_intensity=args.blur)

    if args.input.lower() == 'webcam':
        run_webcam(anonymizer)
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    suffix = input_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        print(f"Error: Unsupported file type '{suffix}'.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else \
        input_path.with_name(f"anonymized_{input_path.name}")

    try:
        if suffix in VIDEO_EXTENSIONS:
            print(f"Processing video. Saving to {output_path}...")
            anonymizer.process_video_file(input_path, output_path)
        else:
            print("Processing image...")
            anonymizer.process_image_file(input_path, output_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved anonymized output to {output_path}")


if __name__ == "__main__":
    main()
