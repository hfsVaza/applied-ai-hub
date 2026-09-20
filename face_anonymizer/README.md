# Occlusion-Aware Face Anonymizer 🎭

A privacy-preserving computer vision mini-application that detects and
anonymizes human faces in images, videos, and live webcam streams.

Unlike a standard bounding-box blur, this application has a **perfect** mode
that uses AI face parsing (semantic segmentation) to blur only the face pixels
themselves. If something is in front of the face — a hand, a microphone, a
coffee cup — that object stays perfectly sharp while the face behind it is
destroyed.

## Features
* **Inputs:** still images (`.jpg`, `.png`, `.bmp`, `.webp`), video files
  (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`), and a live webcam.
* **Modes:**
  * `basic` — bounding-box blur. Very fast, but blurs anything inside the box.
  * `perfect` — pixel-accurate semantic mask that respects occlusions.
* **Robustness:** handles multiple faces at once, and returns frames with no
  faces untouched rather than crashing.
* **Batched segmentation:** all faces in a frame are segmented in a single
  forward pass, so a group photo costs little more than a portrait.
* **Live toggle:** press `m` during the webcam demo to flip between `basic`
  and `perfect` and compare them directly.

## Setup

### Prerequisites
* Python 3.9+
* A reasonable CPU, or Apple Silicon (MPS), or an Nvidia GPU (CUDA) for
  acceptable speed in `perfect` mode.

### Installation
```bash
cd face_anonymizer
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
The first `perfect`-mode run downloads the face parsing weights from
HuggingFace (~150MB). The face *detector* (`blaze_face_short_range.tflite`) is
bundled with this folder.

## Usage

### 1. Live webcam demo
```bash
python main.py
```
* `m` toggles between `perfect` and `basic`. Try holding your hand in front of
  your face and pressing it.
* `q` quits.

### 2. Anonymize an image
```bash
python main.py --input my_photo.jpg
python main.py --input my_photo.jpg --output safe.jpg --blur 91
```

### 3. Anonymize a video
```bash
python main.py --input my_video.mp4
```

Output defaults to `anonymized_<input name>` beside the input file.

---

## Technical Report & Implementation Details

### Models and libraries used
A hybrid two-stage pipeline, for both speed and precision:

1. **Face detection (MediaPipe).** `mediapipe.tasks.vision.FaceDetector` runs a
   lightweight BlazeFace model over the whole frame and returns a bounding box
   per face. This is cheap enough to run on every frame in real time.
2. **Face parsing (Transformers / PyTorch).** A
   `SegformerForSemanticSegmentation` model (`jonathandinu/face-parsing`)
   classifies every pixel into one of 18 categories. Crucially it runs only on
   the *cropped boxes* from stage 1, not the whole image — the reason the
   pipeline is fast enough to be usable at all.

The mask is built from the classes that carry identity: skin, nose, eyes,
brows, mouth, lips, and eyeglasses. Hair is deliberately **excluded**: it does
not identify a person the way facial features do, and blurring it produces an
obvious smeared halo around the head.

### Anonymization method and irreversibility
The method is a **Gaussian blur** applied through the semantic mask.

**Is it truly irreversible?** Not in the strict cryptographic sense, and it is
important not to overclaim here. A Gaussian blur is a convolution, and a small
kernel is partially invertible by deconvolution; modern generative models can
also hallucinate plausible faces from blurred input. What a large kernel (51+
by default) does is destroy the high-frequency detail that both human and
algorithmic recognition depend on, well past the point of practical recovery.
For genuinely adversarial threat models, replacing the region with solid pixels
or a synthetic face is the stronger choice. Blur is chosen here because it
keeps the video watchable, which is the actual use case.

### Handling multiple faces and no-face frames
* **Multiple faces:** all detections are collected, cropped, and segmented in
  one batched forward pass, then blended back individually.
* **No faces:** an empty detection list returns the original frame immediately,
  avoiding both a crash and a pointless copy.

### Inputs that work well vs. limitations
* **Works well:** well-lit scenes with clearly visible faces, including ones
  partially occluded by hands, microphones, or accessories. `perfect` mode
  shines exactly where box blur fails — people eating, gesturing, or holding
  objects near the face.
* **Limitations:**
  * **Extreme profiles.** BlazeFace is tuned for frontal and near-frontal
    faces. At a full 90° profile the detector can lose the face entirely, and
    an undetected face is an unblurred face.
  * **Small or distant faces.** Faces occupying few pixels fall below the
    detector's confidence threshold.
  * **Performance.** Segmentation is heavy. Cropping first helps enormously,
    but high-resolution video without a GPU or MPS is still slow.
  * **Mask bleed.** The parser sometimes misreads a jawline or hairline edge.
    The mask is feathered with a Gaussian blur before blending so the
    transition is soft rather than a hard-edged cutout.

### Privacy and ethical considerations
* **Local processing.** Everything runs locally; no image or video frame is
  ever transmitted. This aligns with GDPR data-minimisation principles.
* **False negatives are the real risk.** A false positive — blurring a
  background object — is merely annoying. A false negative is a privacy breach,
  and it only takes one. Motion blur, a sudden lighting change, or a turned
  head can drop a frame. **Processed video must be reviewed before release; do
  not treat this tool as a guarantee.**
* **Detector bias.** Face detection accuracy is not uniform across skin tones,
  ages, and face shapes, because of imbalances in training data. In a privacy
  tool this is a fairness problem with direct consequences: the people the
  detector fails on are the people who go unprotected.

### Sources & documentation
* [MediaPipe Face Detection](https://developers.google.com/mediapipe/solutions/vision/face_detector)
* [HuggingFace Segformer](https://huggingface.co/docs/transformers/model_doc/segformer)
* [OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
* Course reading: "Face Detection and Image/Video Anonymization"
