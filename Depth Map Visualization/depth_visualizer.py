"""Monocular depth map estimation and visualisation.

Runs a Hugging Face depth-estimation pipeline over images or video and renders
the result as a grayscale or colour-mapped map, optionally side by side with
the original, or exported as 3D geometry.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import pipeline
import cv2
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_CHOICES = [
    "depth-anything/Depth-Anything-V2-Small-hf",
    "Intel/dpt-large",
    "Intel/dpt-hybrid-midas",
]
COLORMAP_CHOICES = ["gray", "inferno", "magma", "plasma", "viridis", "turbo"]

# Vertices per side in the exported mesh. The browser viewer bogs down well
# before a full-resolution grid, so the mesh is downsampled to this dimension.
MESH_MAX_DIM = 256


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


def load_image(image_path):
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}", file=sys.stderr)
        return None


def normalize_depth(depth_array, invert=False):
    """Scale a raw depth prediction to the 0-255 range for display.

    The models output relative depth on an arbitrary scale, so the only
    meaningful mapping is min-to-max within this one frame. That is also why
    the same colour in two different images is not the same distance.
    """
    d_min, d_max = np.min(depth_array), np.max(depth_array)

    if d_max == d_min:
        normalized = np.zeros_like(depth_array, dtype=np.uint8)
    else:
        normalized = (255.0 * (depth_array - d_min) / (d_max - d_min)).astype(np.uint8)

    return 255 - normalized if invert else normalized


def depth_to_distance(depth_array):
    """Convert a raw prediction into a 0-1 distance where 0 is nearest.

    All three supported models predict *inverse* depth: a larger value means
    the pixel is closer to the camera. Geometry exports need the opposite
    convention, so the normalised map is flipped here. Without this the 3D
    exports come out turned inside out, with foreground objects pushed
    furthest from the viewer.
    """
    d_min, d_max = float(np.min(depth_array)), float(np.max(depth_array))
    if d_max == d_min:
        return np.zeros_like(depth_array, dtype=np.float32)
    normalized = (depth_array - d_min) / (d_max - d_min)
    return (1.0 - normalized).astype(np.float32)


def apply_colormap(normalized_depth, colormap_name="gray"):
    if colormap_name == "gray":
        return Image.fromarray(normalized_depth, mode="L").convert("RGB")

    try:
        cmap = plt.get_cmap(colormap_name)
        colorized = cmap(normalized_depth / 255.0)
        return Image.fromarray((colorized[:, :, :3] * 255).astype(np.uint8))
    except ValueError:
        return Image.fromarray(normalized_depth, mode="L").convert("RGB")


def apply_colormap_with_legend(normalized_depth, colormap_name="gray"):
    """Colour-map the depth and render a matplotlib colourbar beside it."""
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.imshow(normalized_depth, cmap=colormap_name, vmin=0, vmax=255)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relative depth (0 = far, 255 = near)')
    ax.axis('off')

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba).convert('RGB')
    plt.close(fig)
    return img


def create_side_by_side(original_image, depth_image):
    if original_image.size != depth_image.size:
        depth_image = depth_image.resize(original_image.size, Image.Resampling.LANCZOS)

    w, h = original_image.size
    sbs_image = Image.new('RGB', (w * 2, h))
    sbs_image.paste(original_image, (0, 0))
    sbs_image.paste(depth_image, (w, 0))
    return sbs_image


def export_ply(original_image, depth_array, output_path):
    """Export a coloured 3D point cloud as a binary PLY file.

    Written as binary rather than ASCII text: a 1080p frame is two million
    points, which as text takes minutes to write and tens of megabytes on
    disk. Every viewer that reads PLY (MeshLab, CloudCompare, Open3D, Brush)
    reads the binary form.
    """
    w, h = original_image.size
    distance = depth_to_distance(depth_array)
    distance = cv2.resize(distance, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = np.array(original_image)

    # Back-project pixels through a pinhole camera to get real 3D positions.
    # Without dividing by the focal length the cloud stays a flat billboard
    # instead of spreading out with distance.
    focal_length = max(w, h)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = distance * focal_length
    x = (xs - w / 2) * z / focal_length
    y = -(ys - h / 2) * z / focal_length

    vertices = np.empty(
        h * w,
        dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')],
    )
    vertices['x'] = x.ravel()
    vertices['y'] = y.ravel()
    vertices['z'] = z.ravel()
    vertices['red'] = rgb[:, :, 0].ravel()
    vertices['green'] = rgb[:, :, 1].ravel()
    vertices['blue'] = rgb[:, :, 2].ravel()

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.size}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertices.tobytes())

    return str(output_path)


def export_mesh_ply(image, depth_map, output_path="mesh.ply"):
    """Export a triangulated depth mesh for the in-browser 3D viewer."""
    rgb = np.array(image.convert("RGB"))
    distance = depth_to_distance(depth_map)
    h, w = distance.shape

    # Downsample so the browser viewer stays responsive.
    if max(h, w) > MESH_MAX_DIM:
        scale = MESH_MAX_DIM / max(h, w)
        w, h = int(w * scale), int(h * scale)
        distance = cv2.resize(distance, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

    # Triangulate the grid, dropping quads that straddle a depth discontinuity.
    # Those quads are the boundary between a foreground object and the distant
    # background, and connecting them smears the object into a rubber sheet.
    idx = np.arange(h * w).reshape(h, w)
    i1, i2, i3, i4 = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    d1, d2, d3, d4 = distance[:-1, :-1], distance[:-1, 1:], distance[1:, :-1], distance[1:, 1:]

    depth_thresh = 0.05 * float(np.ptp(distance))
    spread = np.maximum(np.maximum(np.abs(d1 - d2), np.abs(d1 - d3)), np.abs(d1 - d4))
    valid = spread < depth_thresh if depth_thresh > 0 else np.ones_like(spread, dtype=bool)

    faces = np.concatenate([
        np.stack([i1[valid], i2[valid], i3[valid]], axis=1),
        np.stack([i2[valid], i4[valid], i3[valid]], axis=1),
    ])

    # Centre the mesh on the origin and scale to roughly a unit cube, so it
    # arrives in the viewer already framed rather than needing a manual zoom.
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    verts = np.column_stack([
        (xs / w - 0.5).ravel(),
        (-(ys / h) + 0.5).ravel(),
        (distance * 2.0).ravel(),
        rgb[:, :, 0].ravel(),
        rgb[:, :, 1].ravel(),
        rgb[:, :, 2].ravel(),
    ])

    with open(output_path, 'w') as f:
        f.write(
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {h * w}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_index\n"
            "end_header\n"
        )
        np.savetxt(f, verts, fmt="%.5f %.5f %.5f %d %d %d")
        np.savetxt(f, np.column_stack([np.full(len(faces), 3), faces]), fmt="%d")

    return str(output_path)


def get_model_pipeline(model_id):
    """Load a depth-estimation pipeline on the best available accelerator."""
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1
    return pipeline("depth-estimation", model=model_id, device=device)


def _predict_depth(pipe, pil_image):
    """Run the pipeline and return the raw depth prediction as a 2D array.

    The pipeline hands back a torch tensor. Converting it with np.array()
    directly trips a NumPy 2 deprecation over the `copy` keyword, so the
    tensor is detached and moved to the CPU explicitly first.
    """
    depth = pipe(pil_image)["predicted_depth"].squeeze()
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    return np.asarray(depth, dtype=np.float32)


def _render_depth_frame(raw_depth, source_image, colormap, invert, side_by_side, add_legend):
    """Turn a raw prediction into the final visualisation image."""
    normalized_depth = normalize_depth(raw_depth, invert=invert)

    if add_legend:
        depth_image = apply_colormap_with_legend(normalized_depth, colormap_name=colormap)
    else:
        depth_image = apply_colormap(normalized_depth, colormap_name=colormap)
        depth_image = depth_image.resize(source_image.size, Image.Resampling.LANCZOS)

    if not side_by_side:
        return depth_image

    # The legend changes the aspect ratio, so match heights before stacking.
    if add_legend:
        new_w = int(depth_image.width * (source_image.height / depth_image.height))
        depth_image = depth_image.resize((new_w, source_image.height))
    return create_side_by_side(source_image, depth_image)


def process_video(pipe, input_path, output_dir, colormap, invert, side_by_side,
                  add_legend=False, progress=None):
    """Estimate depth for every frame of a video and write an annotated copy."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        base_name = Path(input_path).stem
        suffix = "_sbs" if side_by_side else "_depth"
        output_path = os.path.join(output_dir, f"{base_name}{suffix}_{colormap}.mp4")

        writer = None
        frame_index = 0
        try:
            for _ in tqdm(range(total_frames) if total_frames > 0 else iter(int, 1),
                          desc=f"Depth: {base_name}"):
                ret, frame = cap.read()
                if not ret:
                    break

                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                raw_depth = _predict_depth(pipe, pil_image)
                final_image = _render_depth_frame(
                    raw_depth, pil_image, colormap, invert, side_by_side, add_legend
                )
                final_bgr = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)

                if writer is None:
                    # Size the writer from the real first frame, since the
                    # legend and side-by-side options both change dimensions.
                    h_out, w_out = final_bgr.shape[:2]
                    writer = open_video_writer(output_path, fps, (w_out, h_out))
                    expected = (h_out, w_out)
                elif final_bgr.shape[:2] != expected:
                    final_bgr = cv2.resize(final_bgr, (expected[1], expected[0]))

                writer.write(final_bgr)

                frame_index += 1
                if progress is not None and total_frames > 0:
                    progress(
                        min(frame_index / total_frames, 1.0),
                        f"Depth: frame {frame_index}/{total_frames}",
                    )
        finally:
            if writer is not None:
                writer.release()
    finally:
        cap.release()

    if writer is None:
        raise ValueError(f"No frames could be read from {input_path}")

    print(f"Saved video to: {output_path}")
    return output_path


def process_image(pipe, input_path, output_dir, colormap, invert, side_by_side,
                  export_point_cloud=False, export_npy=False, export_mesh=False,
                  add_legend=False):
    """Estimate depth for one image. Returns (visualisation, extra_files)."""
    image = load_image(input_path)
    if image is None:
        return None, []

    try:
        raw_depth = _predict_depth(pipe, image)
    except Exception as e:
        print(f"Error during model inference for '{input_path}': {e}", file=sys.stderr)
        return None, []

    final_image = _render_depth_frame(
        raw_depth, image, colormap, invert, side_by_side, add_legend
    )

    base_name = Path(input_path).stem
    suffix = "_sbs" if side_by_side else "_depth"
    output_path = os.path.join(output_dir, f"{base_name}{suffix}_{colormap}.png")

    try:
        final_image.save(output_path)
        print(f"Saved visualization to: {output_path}")
    except Exception as e:
        print(f"Error saving output for '{input_path}': {e}", file=sys.stderr)
        return None, []

    generated_files = []
    if export_point_cloud:
        generated_files.append(
            export_ply(image, raw_depth, os.path.join(output_dir, f"{base_name}_pointcloud.ply"))
        )
    if export_mesh:
        generated_files.append(
            export_mesh_ply(image, raw_depth, os.path.join(output_dir, f"{base_name}_mesh.ply"))
        )
    if export_npy:
        npy_path = os.path.join(output_dir, f"{base_name}_raw_depth.npy")
        np.save(npy_path, raw_depth)
        generated_files.append(npy_path)

    return output_path, generated_files


def main():
    parser = argparse.ArgumentParser(description="Monocular Depth Map Visualizer")
    parser.add_argument("--input", "-i", required=True, help="Input image, video, or directory")
    parser.add_argument("--output", "-o", default="output", help="Output directory path")
    parser.add_argument("--colormap", "-c", default="inferno", choices=COLORMAP_CHOICES,
                        help="Colormap for the depth visualisation")
    parser.add_argument("--invert", action="store_true", help="Swap near and far in the colours")
    parser.add_argument("--sbs", action="store_true", help="Show original and depth side by side")
    parser.add_argument("--ply", action="store_true", help="Export a 3D point cloud (.ply)")
    parser.add_argument("--mesh", action="store_true", help="Export a 3D triangle mesh (.ply)")
    parser.add_argument("--npy", action="store_true", help="Export the raw depth array (.npy)")
    parser.add_argument("--legend", action="store_true", help="Add a colorbar legend")
    parser.add_argument("--model", default=MODEL_CHOICES[0], help="Hugging Face model ID")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading depth estimation model: {args.model}")
    pipe = get_model_pipeline(args.model)

    if input_path.is_file():
        targets = [input_path]
    else:
        targets = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in (IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
        )
        if not targets:
            print(f"Error: No images or videos found in '{input_path}'.", file=sys.stderr)
            sys.exit(1)

    for target in targets:
        try:
            if target.suffix.lower() in VIDEO_EXTENSIONS:
                process_video(pipe, target, output_dir, args.colormap, args.invert,
                              args.sbs, args.legend)
            else:
                process_image(pipe, target, output_dir, args.colormap, args.invert,
                              args.sbs, args.ply, args.npy, args.mesh, args.legend)
        except Exception as e:
            print(f"Error processing '{target}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
