# Monocular Depth Map Visualizer

A Python mini-application for estimating and visualising depth from ordinary 2D
images and videos. This project fulfils the "Depth Map Visualization" learning
goals and practical requirements for the Applied AI Programming course.

## Overview

The application takes a 2D image (JPG, PNG) or video (MP4, MOV) and produces a
depth map: an image where each pixel's value encodes estimated relative depth.

It uses **Depth Anything V2 (Small)** through the HuggingFace `transformers`
library by default, and can switch to Intel's DPT/MiDaS models from the UI or
the `--model` flag.

## Features
- **Image and video support**, processed frame by frame.
- **Web UI** built with Gradio, plus a full command-line interface.
- **Normalisation** of raw predictions into a 0–255 displayable range.
- **Grayscale and colour-mapped output** (`inferno`, `magma`, `plasma`,
  `viridis`, `turbo`) to make small depth differences visible.
- **Side-by-side comparison** of the original and its depth map.
- **Colorbar legend** rendered onto the image.
- **3D point cloud export** (`.ply`) for use in MeshLab, CloudCompare, Open3D,
  or Gaussian splat tools such as Brush.
- **3D mesh export** (`.ply`, with faces) viewable directly in the browser.
- **Raw data export** (`.npy`) for numerical analysis.
- **Batch processing** of a whole folder from the CLI, or multiple uploads in
  the UI.
- **Invert support** to swap the near/far colour mapping.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

## How to Run

### Web interface
```bash
python app.py          # standalone, on http://127.0.0.1:7861
python ../hub.py       # or as part of the combined hub, on port 7860
```

### Command line
```bash
# Basic grayscale
python depth_visualizer.py -i input.jpg -c gray

# Colorized, side by side, with a colorbar legend
python depth_visualizer.py -i input.jpg -c inferno --sbs --legend

# Batch a folder, invert the mapping, export point clouds
python depth_visualizer.py -i my_images_folder -c magma --invert --sbs --ply

# Export a browser-viewable 3D mesh
python depth_visualizer.py -i input.jpg --mesh

# Process a video
python depth_visualizer.py -i my_video.mp4 --sbs
```

## Project Report

### 1. What depth estimation model, library, API, or method was used?
**Depth Anything V2** (`depth-anything/Depth-Anything-V2-Small-hf`) by default,
loaded through the HuggingFace `transformers` `pipeline` API with PyTorch as the
backend. The UI and `--model` flag also offer `Intel/dpt-large` and
`Intel/dpt-hybrid-midas`. Inference runs on Apple MPS or CUDA when available and
falls back to CPU otherwise.

### 2. Relative depth vs. metric depth
These models predict **relative depth**. The map reliably orders objects by
distance within a single scene, but the values do not correspond to real-world
units such as metres. Two different images are normalised independently, so the
same shade in each does not represent the same physical distance.

All three supported models in fact predict **inverse** depth: a *larger* raw
value means a *closer* pixel. This matters for the 3D exports, which need the
opposite convention. `depth_to_distance()` performs that flip explicitly. An
earlier version of the point cloud exporter did not, which turned every scene
inside out, pushing foreground objects furthest from the viewer.

### 3. Depth normalisation
The model returns a 2D array of floating-point values on an unknown scale. To
visualise them:
1. Find the minimum (`d_min`) and maximum (`d_max`) in the array.
2. Scale linearly to 0.0–1.0.
3. Multiply by 255 and cast to `uint8`.
4. With the invert option, flip it (`255 - value`).

If every pixel has the same value (`d_max == d_min`) the map is returned as
all zeros instead of dividing by zero.

Because the scaling is per-image, normalisation is what makes the map
*displayable* but also what makes it *non-comparable* between images.

### 4. Colour mapping
The 0–255 values are mapped onto a `matplotlib` gradient. The default is
**`inferno`**; `gray`, `magma`, `plasma`, `viridis` and `turbo` are also
available. Since the models predict inverse depth, bright means **near** and
dark means **far** by default; the invert option swaps that. The optional
colorbar legend runs from 0 to 255 and has no units, because the values are
relative.
- **Grayscale** is best for numerical inspection and reading raw structure.
- **Colorized** is far better for human inspection: the eye resolves small
  changes in hue much more readily than small changes in brightness, so subtle
  depth steps that vanish in grayscale become obvious in `inferno` or `turbo`.

### 5. How the 3D exports work
- **Point cloud (`.ply`)** back-projects every pixel through a simple pinhole
  camera model, using `max(width, height)` as a stand-in focal length. Without
  dividing by focal length the result stays a flat billboard rather than
  spreading out with distance. The file is written as **binary** PLY: a 1080p
  frame is 2.07 million points, which as ASCII text is both far slower to write
  and several times larger on disk.
- **Mesh (`.ply`)** triangulates the depth grid, downsampled to 256×256 so the
  browser viewer stays responsive. Quads that straddle a large depth
  discontinuity are **dropped** rather than triangulated: those are the
  boundary between a foreground object and the distant background, and
  connecting them smears the object into a rubber sheet stretching back into
  the scene.

### 6. What works well vs. difficult images
- **Works well:** scenes with clear occlusion, strong texture, a distinct
  foreground, and outdoor landscapes.
- **Difficult:**
  - **Reflective surfaces and mirrors:** the model estimates depth *into* the
    reflection rather than to the surface of the mirror.
  - **Transparent objects** (windows, glass): depth becomes ambiguous between
    the glass and what is behind it.
  - **Very flat, untextured scenes** such as a blank wall: with no cues to work
    from, the output is mostly noise.
  - **Thin structures** such as railings or hair, which fall below the model's
    effective resolution and get absorbed into the background.

### 7. Limitations
- **Video frames are normalised one at a time.** Each frame is scaled by its
  own minimum and maximum, so the same wall can change shade from frame to
  frame when something nearer or farther enters the shot.
- **No real scale in the 3D exports.** The point cloud and mesh inherit the
  relative depth, and the focal length is a stand-in (`max(width, height)`)
  rather than the real camera's. Their shape is plausible, but their
  proportions and distances are not measurements.
- **The mesh is coarse.** It is downsampled to 256×256 vertices, so fine
  detail visible in the depth image is lost in the mesh.
- **Speed.** Every frame is a full model pass, so long videos take a long time
  without a GPU. The Small model is the default because it is by far the
  smallest of the three.

### 8. Privacy, Safety, and Misuse
**Misuse as metric data.** The main safety concern is treating relative depth as
if it were metric. Used in an autonomous vehicle or a robot without LiDAR or
stereo validation, these values could produce badly wrong stopping distances.
The model has no way to signal that it is uncertain, so a confident-looking
depth map is not evidence of a correct one.

**Privacy.** Inference is local, so images are never sent to a cloud service.
Users should still be careful processing images of other people in private
settings: a depth map, and especially an exported point cloud, reveals the 3D
structure and layout of an interior far more completely than the original flat
photograph does.
