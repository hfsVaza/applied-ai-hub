"""Standalone Gradio UI for the depth map visualizer.

This is the single-project version. The combined interface for all four
projects lives in `hub.py` at the repository root.
"""

import traceback
from pathlib import Path

import gradio as gr

from depth_visualizer import (
    COLORMAP_CHOICES,
    MODEL_CHOICES,
    VIDEO_EXTENSIONS,
    get_model_pipeline,
    process_image,
    process_video,
)

OUTPUT_DIR = Path("output")

# Gradio 6 moved theme from the Blocks constructor to launch(); see hub.py.
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
STYLE_KWARGS = {"theme": gr.themes.Soft()}

# Pipelines are cached so switching options does not reload the model.
_pipelines = {}


def get_pipe(model_name):
    if model_name not in _pipelines:
        _pipelines[model_name] = get_model_pipeline(model_name)
    return _pipelines[model_name]


def process_media(input_files, model_choice, colormap, invert, side_by_side,
                  add_legend, export_ply, export_npy, export_mesh,
                  progress=gr.Progress()):
    if not input_files:
        return [], [], None, "Upload at least one image or video first."

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.0, f"Loading {model_choice}...")
        pipe = get_pipe(model_choice)
    except Exception as e:
        traceback.print_exc()
        return [], [], None, f"Could not load the model: {e}"

    output_images, all_output_files, mesh_output, errors = [], [], None, []

    for i, file_obj in enumerate(input_files):
        input_path = Path(file_obj)
        progress(i / len(input_files), f"Processing {input_path.name}...")
        try:
            if input_path.suffix.lower() in VIDEO_EXTENSIONS:
                video_path = process_video(
                    pipe=pipe, input_path=input_path, output_dir=OUTPUT_DIR,
                    colormap=colormap, invert=invert, side_by_side=side_by_side,
                    add_legend=add_legend, progress=progress,
                )
                if video_path:
                    all_output_files.append(video_path)
            else:
                image_path, extra_files = process_image(
                    pipe=pipe, input_path=input_path, output_dir=OUTPUT_DIR,
                    colormap=colormap, invert=invert, side_by_side=side_by_side,
                    export_point_cloud=export_ply, export_npy=export_npy,
                    export_mesh=export_mesh, add_legend=add_legend,
                )
                if image_path:
                    output_images.append(image_path)
                    all_output_files.append(image_path)
                for f in extra_files:
                    all_output_files.append(f)
                    if f.endswith("_mesh.ply"):
                        mesh_output = f
        except Exception as e:
            traceback.print_exc()
            errors.append(f"{input_path.name}: {e}")

    if errors:
        status = f"Finished with {len(errors)} error(s): " + "; ".join(errors)
    else:
        status = f"Done. {len(all_output_files)} file(s) written to `output/`."
    return output_images, all_output_files, mesh_output, status


def build_ui():
    blocks_style = {} if GRADIO_MAJOR >= 6 else STYLE_KWARGS
    with gr.Blocks(title="Monocular Depth Map Visualizer", **blocks_style) as app:
        gr.Markdown(
            "# Monocular Depth Map Visualizer\n"
            "Upload one or more images or videos to estimate relative depth. "
            "Depth *order* is reliable; the values are not real-world distances."
        )

        with gr.Row():
            with gr.Column():
                input_media = gr.File(label="Image(s) or video(s)",
                                      file_types=["image", "video"], file_count="multiple")
                model_choice = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[0],
                                           label="Depth estimation model")
                colormap = gr.Dropdown(COLORMAP_CHOICES, value="inferno", label="Colormap")

                with gr.Row():
                    invert = gr.Checkbox(label="Invert depth", value=False)
                    side_by_side = gr.Checkbox(label="Side-by-side", value=True)
                    add_legend = gr.Checkbox(label="Colorbar legend", value=False)
                with gr.Row():
                    export_ply = gr.Checkbox(label="Point cloud (.ply)", value=False)
                    export_npy = gr.Checkbox(label="Raw depth (.npy)", value=False)
                    export_mesh = gr.Checkbox(label="3D mesh (images only)", value=False)

                process_btn = gr.Button("Generate Depth Maps", variant="primary")

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Images & Videos"):
                        output_gallery = gr.Gallery(label="Preview", columns=2)
                    with gr.Tab("3D Viewer"):
                        output_3d = gr.Model3D(label="Interactive depth mesh")
                output_files = gr.File(label="All outputs (downloads)",
                                       file_count="multiple", interactive=False)
                status = gr.Markdown()

        process_btn.click(
            process_media,
            [input_media, model_choice, colormap, invert, side_by_side,
             add_legend, export_ply, export_npy, export_mesh],
            [output_gallery, output_files, output_3d, status],
        )

    return app


if __name__ == "__main__":
    launch_style = STYLE_KWARGS if GRADIO_MAJOR >= 6 else {}
    build_ui().launch(server_name="127.0.0.1", server_port=7861,
                      inbrowser=True, **launch_style)
