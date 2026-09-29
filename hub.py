"""AI Projects Hub.

A single Gradio front end for four computer-vision and NLP mini-projects:
face anonymization, monocular depth estimation, exercise rep counting, and
semantic document search.

Live webcam work runs on the main thread rather than inside a Gradio callback.
macOS requires all `cv2.imshow` windows to be driven from the main thread, so
the UI buttons only flip flags in `hub_state`, and the loop at the bottom of
this file picks them up and owns the camera.
"""

import os
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_ANON_DIR = os.path.join(BASE_DIR, "face_anonymizer")
DEPTH_MAP_DIR = os.path.join(BASE_DIR, "Depth Map Visualization")
POSE_DIR = os.path.join(BASE_DIR, "pose_estimator")
SEMANTIC_DIR = os.path.join(BASE_DIR, "semantic_search")

for path in (FACE_ANON_DIR, DEPTH_MAP_DIR, POSE_DIR, SEMANTIC_DIR):
    sys.path.append(path)

from depth_visualizer import (  # noqa: E402
    COLORMAP_CHOICES,
    MODEL_CHOICES,
    get_model_pipeline,
    process_image as depth_process_image,
    process_video as depth_process_video,
)
from main import LIVE_PARSE_SIZE, MAX_PARSE_SIZE, FaceAnonymizer  # noqa: E402
from pose_counter import EXERCISE_NAMES, PoseProcessor, process_video_file as pose_process_video  # noqa: E402
from search_engine import SEARCH_METHODS, build_default_engine  # noqa: E402
from file_library import (  # noqa: E402
    DEFAULT_CLUSTER_DISTANCE,
    LIBRARY_DIRNAME,
    FileLibrary,
    supported_extensions,
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Where to listen, and whether this process has a display and a camera to use.
# Inside a container there is neither, so HUB_HEADLESS=1 swaps the native
# webcam windows for in-browser streaming: the browser owns the camera and
# sends frames to the server, which sends the processed frames back.
HUB_HOST = os.environ.get("HUB_HOST", "127.0.0.1")
HUB_PORT = int(os.environ.get("HUB_PORT", "7860"))
HEADLESS = os.environ.get("HUB_HEADLESS", "").lower() in ("1", "true", "yes")
OPEN_BROWSER = not HEADLESS and os.environ.get("HUB_OPEN_BROWSER", "1").lower() not in ("0", "false", "no")

NO_CAMERA_MESSAGE = (
    "Live webcam is unavailable in headless mode. Upload an image or a video "
    "instead, or run the hub directly on your machine."
)

WEB_DIR = Path(BASE_DIR) / "web"

LIBRARY_DIR = Path(SEMANTIC_DIR) / "data" / LIBRARY_DIRNAME

OUTPUT_DIR = Path(BASE_DIR) / "output"
ANON_OUTPUT_DIR = OUTPUT_DIR / "anonymizer"
DEPTH_OUTPUT_DIR = OUTPUT_DIR / "depth"
POSE_OUTPUT_DIR = OUTPUT_DIR / "pose"
for directory in (ANON_OUTPUT_DIR, DEPTH_OUTPUT_DIR, POSE_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def unique_output_path(directory, stem, suffix):
    """Build a timestamped output path.

    Every run previously wrote to one fixed filename, so a second run silently
    destroyed the first one's result and two browser tabs would fight over the
    same file. Stamping the name keeps every run's output intact.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_stem = "".join(c for c in stem if c.isalnum() or c in "-_ ").strip() or "output"
    return str(Path(directory) / f"{safe_stem}_{stamp}{suffix}")


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


# ------------------------------------------------------- native window help

# Keys that close a live window. 255 is what `waitKey(1) & 0xFF` returns when
# no key was pressed.
QUIT_KEYS = {ord('q'), ord('Q'), 27}   # 27 = Esc
NO_KEY = 255

ANON_WINDOW = "Live Anonymizer - press 'q' to quit"
POSE_WINDOW = "Live Rep Counter - press 'q' to quit"


def pump_gui(cycles=2):
    """Service OpenCV's window event loop and return any key that was pressed.

    macOS needs the Cocoa event queue drained regularly, or it marks the window
    unresponsive, shows the spinning beachball, and discards queued key
    presses. One `waitKey(1)` per frame is not enough when the frame itself
    costs 100ms+ of model inference: measured, the window was serviced 0.75% of
    the time, which is why 'q' did nothing and the window froze.

    Each `waitKey(1)` costs about 12ms of real event processing on macOS, so
    two per frame gives the window roughly a quarter of the frame budget while
    still leaving ~11 FPS of throughput.
    """
    key = NO_KEY
    for _ in range(cycles):
        pressed = cv2.waitKey(1) & 0xFF
        if pressed != NO_KEY:
            key = pressed
    return key


def window_is_closed(name):
    """True once the window has gone, including via its red close button."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def close_window(name):
    """Close a live window and make sure macOS really removes it.

    `destroyWindow` only marks the window for teardown; the underlying Cocoa
    window stays on screen until the event loop runs again. Without this
    pumping afterwards it lingers as a frozen ghost that beachballs, because
    by then the main thread has moved on and nothing is servicing it.
    """
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass
    cv2.destroyAllWindows()
    for _ in range(10):
        cv2.waitKey(1)


def open_camera():
    """Open the default camera at a resolution that keeps live work fast."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


# --------------------------------------------------------------- lazy models

# The hub front-loads nothing. Each model is built the first time its project
# is actually used, so the UI is reachable in a second or two instead of after
# every model in the suite has been downloaded and moved onto the GPU.
_models = {}


def get_face_anonymizer(mode):
    """Return a cached FaceAnonymizer, falling back to basic mode on failure."""
    key = f"anon_{mode}"
    if key not in _models:
        try:
            log_event(f"loading face anonymizer ({mode})")
            _models[key] = FaceAnonymizer(mode=mode, blur_intensity=51)
            log_event(f"face anonymizer ({mode}) ready", "ok")
        except Exception as e:
            log_event(f"face anonymizer ({mode}) failed: {e}", "error")
            if mode == "perfect":
                return get_face_anonymizer("basic")
            raise
    return _models[key]


def get_depth_pipe(model_name):
    key = f"depth_{model_name}"
    if key not in _models:
        log_event(f"loading depth model {model_name.split('/')[-1]}")
        _models[key] = get_model_pipeline(model_name)
        log_event(f"depth model {model_name.split('/')[-1]} ready", "ok")
    return _models[key]


def get_embedding_model():
    """One SentenceTransformer shared by the search index and the library."""
    if "embedder" not in _models:
        from sentence_transformers import SentenceTransformer
        from search_engine import DEFAULT_MODEL
        log_event("loading embedding model all-MiniLM-L6-v2")
        _models["embedder"] = SentenceTransformer(DEFAULT_MODEL)
        log_event("embedding model ready", "ok")
    return _models["embedder"]


def get_search_engine(rebuild=False):
    if rebuild:
        _models.pop("search", None)
    if "search" not in _models:
        engine = build_default_engine(BASE_DIR, model=get_embedding_model())
        _models["search"] = engine
        log_event(f"search index built: {len(engine.documents)} chunks", "ok")
    return _models["search"]


def get_library():
    if "library" not in _models:
        library = FileLibrary(LIBRARY_DIR, model=get_embedding_model())
        count = library.load_existing()
        _models["library"] = library
        log_event(f"file library opened: {count} file(s)")
    return _models["library"]


def configured_anonymizer(mode, blur_intensity, parse_size=MAX_PARSE_SIZE):
    """The cached model is shared by file and live work, so every caller sets
    the segmentation size it wants rather than inheriting the last one used."""
    model = get_face_anonymizer(mode)
    model.blur_intensity = blur_intensity
    model.parse_size = parse_size
    return model


# ----------------------------------------------------------------- event log

# A short rolling history surfaced on the landing page. Real events only:
# a log that invents activity is worse than no log at all.
EVENTS = deque(maxlen=60)
STARTED_AT = time.time()


def log_event(message, level="info"):
    """Record an event for the landing page's process log, and echo it."""
    EVENTS.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "message": message,
    })
    print(f"[{level}] {message}")


# ---------------------------------------------------------------- app state

hub_state = {
    'anon_running': False,
    'anon_mode': 'perfect',
    'anon_blur': 51,

    'pose_running': False,
    'pose_exercise': EXERCISE_NAMES[0],
    'pose_3d': False,
}


# ------------------------------------------------------- project 1: faces

def anon_process_media_ui(input_file, mode, blur_intensity, progress=gr.Progress()):
    """Anonymize an uploaded image or video."""
    if input_file is None:
        return None, None, "Upload an image or a video first."

    stem = Path(input_file).stem
    try:
        progress(0.0, "Loading model...")
        model = configured_anonymizer(mode, blur_intensity)

        if is_video(input_file):
            out_path = unique_output_path(ANON_OUTPUT_DIR, stem, ".mp4")
            model.process_video_file(input_file, out_path, progress=progress)
            return None, out_path, f"Done. Saved to `output/anonymizer/{Path(out_path).name}`"

        out_path = unique_output_path(ANON_OUTPUT_DIR, stem, ".jpg")
        model.process_image_file(input_file, out_path)
        return out_path, None, f"Done. Saved to `output/anonymizer/{Path(out_path).name}`"
    except Exception as e:
        traceback.print_exc()
        return None, None, f"Failed: {e}"


def anon_update_mode(mode):
    hub_state['anon_mode'] = mode
    return f"Mode set to {mode}."


def anon_update_blur(blur):
    hub_state['anon_blur'] = blur
    return f"Blur set to {int(blur)}."


def anon_start_cam(mode, blur):
    if HEADLESS:
        return NO_CAMERA_MESSAGE
    if hub_state['pose_running']:
        return "Stop the rep counter first, both cannot use the camera at once."
    if hub_state['anon_running']:
        return "Already running. Look for the native window."
    hub_state['anon_mode'] = mode
    hub_state['anon_blur'] = blur
    hub_state['anon_running'] = True
    return "Starting native webcam window. Press 'q' in that window to close it."


def anon_stop_cam():
    hub_state['anon_running'] = False
    return "Stopping webcam."


# ------------------------------------------------------- project 2: depth

def depth_process_media_ui(input_files, model_choice, colormap, invert, side_by_side,
                           add_legend, export_ply, export_npy, gen_mesh,
                           progress=gr.Progress()):
    """Generate depth maps for every uploaded image or video."""
    if not input_files:
        return [], [], None, "Upload at least one image or video first."

    try:
        progress(0.0, f"Loading {model_choice}...")
        pipe = get_depth_pipe(model_choice)
    except Exception as e:
        traceback.print_exc()
        return [], [], None, f"Could not load the model: {e}"

    output_images, all_output_files, mesh_output, errors = [], [], None, []

    for i, file_obj in enumerate(input_files):
        input_path = Path(file_obj)
        progress(i / len(input_files), f"Processing {input_path.name}...")
        try:
            if is_video(input_path):
                video_path = depth_process_video(
                    pipe=pipe, input_path=input_path, output_dir=DEPTH_OUTPUT_DIR,
                    colormap=colormap, invert=invert, side_by_side=side_by_side,
                    add_legend=add_legend, progress=progress,
                )
                if video_path:
                    all_output_files.append(video_path)
            else:
                image_path, extra_files = depth_process_image(
                    pipe=pipe, input_path=input_path, output_dir=DEPTH_OUTPUT_DIR,
                    colormap=colormap, invert=invert, side_by_side=side_by_side,
                    export_point_cloud=export_ply, export_npy=export_npy,
                    export_mesh=gen_mesh, add_legend=add_legend,
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
        status = f"Done. {len(all_output_files)} file(s) in `output/depth/`."
        if gen_mesh and mesh_output is None:
            status += " No mesh was produced; meshes are only exported for images."
    return output_images, all_output_files, mesh_output, status


# -------------------------------------------------------- project 3: pose

def pose_process_media_ui(input_file, exercise, show_3d, progress=gr.Progress()):
    """Count reps in an uploaded video."""
    if input_file is None:
        return None, "Upload a video first."
    if not is_video(input_file):
        return None, "That is not a video. Upload an .mp4, .mov, .avi, .mkv or .webm file."

    try:
        progress(0.0, "Loading pose model...")
        out_path = unique_output_path(
            POSE_OUTPUT_DIR, f"{Path(input_file).stem}_{exercise.replace(' ', '')}", ".mp4"
        )
        out_path, reps = pose_process_video(
            input_file, out_path, exercise, show_3d, progress=progress
        )
        return out_path, f"Counted **{reps}** {exercise} rep(s). Saved to `output/pose/{Path(out_path).name}`"
    except Exception as e:
        traceback.print_exc()
        return None, f"Failed: {e}"


def pose_update_exercise(ex):
    hub_state['pose_exercise'] = ex
    return f"Exercise set to {ex}. The counter resets when the exercise changes."


def pose_update_3d(show_3d):
    hub_state['pose_3d'] = show_3d
    return f"3D skeleton viewer: {'on' if show_3d else 'off'}."


def pose_start_cam(exercise, show_3d):
    if HEADLESS:
        return NO_CAMERA_MESSAGE
    if hub_state['anon_running']:
        return "Stop the face anonymizer first, both cannot use the camera at once."
    if hub_state['pose_running']:
        return "Already running. Look for the native window."
    hub_state['pose_exercise'] = exercise
    hub_state['pose_3d'] = show_3d
    hub_state['pose_running'] = True
    return "Starting native tracker window. Press 'q' in that window to close it."


def pose_stop_cam():
    hub_state['pose_running'] = False
    return "Stopping tracker."


# ------------------------------------------------ browser webcam streaming

# In headless mode the browser owns the webcam and streams frames here, so the
# live projects work in a container, where there is no camera to open and no
# display to draw a window on. Gradio frames are RGB; the models expect BGR.
#
# The anonymizer does not use Gradio's streaming webcam: that one needs a
# record click and always shows the raw camera beside the output. Instead a
# small script in GRADIO_HEAD posts JPEG frames to /api/anon/frame and shows
# only what comes back.

STREAM_EVERY = 0.1   # seconds between frames the browser sends (rep counter)


def anon_browser_frame(jpeg_bytes, mode, blur):
    """Anonymize one JPEG frame from the browser's webcam; return a JPEG.

    Returns None if the bytes are not a decodable image.
    """
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    model = configured_anonymizer(mode, blur, parse_size=LIVE_PARSE_SIZE)

    start = time.time()
    output = model.process_frame(frame)
    fps = 1.0 / max(time.time() - start, 1e-6)

    cv2.putText(output, f"{mode} | blur {model.blur_intensity} | {fps:.0f} FPS",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    ok, jpeg = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes() if ok else None


def close_pose_session(session):
    """Release a browser session's landmarker when the session ends."""
    if session is not None:
        session["processor"].close()


def pose_stream_frame(frame, exercise, show_3d, session):
    """Track and count one frame from the browser's webcam.

    Each browser tab keeps its own PoseProcessor in `session`, so two tabs do
    not share a rep count. Changing the exercise rebuilds it, which also
    resets the count: the old count belongs to the old exercise.
    """
    if frame is None:
        return None, session
    if session is None or (session["exercise"], session["show_3d"]) != (exercise, show_3d):
        close_pose_session(session)
        session = {
            "exercise": exercise,
            "show_3d": show_3d,
            "processor": PoseProcessor(exercise=exercise, show_3d=show_3d),
            "fps": None,
        }

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    start = time.time()
    output = session["processor"].process_frame(bgr, fps=session["fps"])
    session["fps"] = 1.0 / max(time.time() - start, 1e-6)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB), session


def pose_stream_reset(session):
    """Zero the count in place. Closing the processor here instead would pull
    it out from under a frame the stream may be processing at this moment."""
    if session is not None:
        session["processor"].counter.reset()
    return "Counter reset."


# ------------------------------------------------------ project 4: search

def perform_search(query, search_method, top_k):
    if not query.strip():
        return "Enter a search query above."

    try:
        engine = get_search_engine()
        results = engine.search(query, method=search_method.lower(), top_k=int(top_k))
    except Exception as e:
        traceback.print_exc()
        return f"Search failed: {e}"

    if not results:
        return (f"No results for *'{query}'*. "
                "Keyword search needs an exact word match, so try Semantic or Hybrid.")

    lines = [f"### {len(results)} result(s) for *'{query}'*  ({search_method})\n"]
    for i, res in enumerate(results, 1):
        meta = res['metadata']
        lines.append(
            f"**{i}. {meta['title']} → {meta['section']}**  \n"
            f"<sub>`{meta['file']}` · chunk {meta['chunk_id']} · score {res['score']:.3f}</sub>\n\n"
            f"> {res['text']}\n\n---\n"
        )
    return "\n".join(lines)


def search_corpus_summary():
    try:
        engine = get_search_engine()
        files = sorted({m['file'] for m in engine.doc_metadata})
        listing = "\n".join(f"- `{f}`" for f in files)
        return f"Indexed **{len(engine.documents)}** chunks from **{len(files)}** documents:\n\n{listing}"
    except Exception as e:
        return f"Could not build the index: {e}"



# ------------------------------------------------- project 4: file library

ORGANIZE_MODES = ["Content similarity", "File type"]


def _render_groups(groups, mode, total):
    """Format grouped files as markdown."""
    if not groups:
        return "The library is empty. Upload some files above to get started."

    heading = ("grouped by what they are **about**" if mode == ORGANIZE_MODES[0]
               else "grouped by what they **are**")
    lines = [f"### {total} file(s), {heading}\n"]

    for label, items in groups:
        lines.append(f"#### {label}  <sub>({len(items)})</sub>\n")
        lines.append("| File | Type | Size | Text |")
        lines.append("|---|---|---:|---|")
        for f in items:
            note = {"ok": "extracted",
                    "empty": "no text found",
                    "described": "name only"}.get(f.status, f.status)
            lines.append(f"| `{f.name}` | {f.category} | {f.size_label} | {note} |")
        lines.append("")
    return "\n".join(lines)


def library_add(files, mode, breadth, progress=gr.Progress()):
    """Copy uploaded files into the library, extract text, and re-index."""
    if not files:
        return "Choose some files first.", library_organize(mode, breadth), library_stats()

    try:
        progress(0.1, "Loading embedding model...")
        library = get_library()
        progress(0.3, f"Extracting text from {len(files)} file(s)...")
        added, skipped = library.add(files)

        progress(0.8, "Re-indexing for search...")
        get_search_engine(rebuild=True)

        message = f"Added **{len(added)}** file(s)."
        unreadable = [f.name for f in added if f.status not in ("ok",)]
        if unreadable:
            message += (f" {len(unreadable)} had no extractable text and are "
                        f"filed by name only: {', '.join(unreadable[:4])}"
                        + ("..." if len(unreadable) > 4 else ""))
        if skipped:
            message += f" Skipped {len(skipped)}: " + ", ".join(n for n, _ in skipped[:4])
        message += " They are now searchable in the Search tab."
        return message, library_organize(mode, breadth), library_stats()
    except Exception as e:
        traceback.print_exc()
        return f"Failed: {e}", "", library_stats()


def library_organize(mode, breadth, progress=gr.Progress()):
    """Group the library either by subject matter or by file type."""
    try:
        library = get_library()
        if not library.files:
            return "The library is empty. Upload some files above to get started."

        if mode == ORGANIZE_MODES[1]:
            return _render_groups(library.group_by_type(), mode, len(library.files))

        progress(0.3, "Embedding documents...")
        groups = library.group_by_content(distance=float(breadth))
        return _render_groups(groups, mode, len(library.files))
    except Exception as e:
        traceback.print_exc()
        return f"Could not organize the library: {e}"


def library_stats():
    try:
        library = get_library()
    except Exception as e:
        return f"Library unavailable: {e}"
    if not library.files:
        return "*Library is empty.*"

    readable = sum(1 for f in library.files if f.status == "ok")
    total_size = sum(f.size for f in library.files) / 1e6
    kinds = len({f.category for f in library.files})
    return (f"**{len(library.files)}** file(s) · {kinds} type(s) · {total_size:.1f} MB · "
            f"{readable} with extractable text")


def library_clear(mode, breadth):
    try:
        removed = get_library().clear()
        get_search_engine(rebuild=True)
        return (f"Removed {removed} file(s) from the library.",
                library_organize(mode, breadth), library_stats())
    except Exception as e:
        traceback.print_exc()
        return f"Failed to clear: {e}", "", library_stats()


# ------------------------------------------------------------------- the UI

# Leaving a project returns to the landing film *at its end*, where the menu
# is, rather than replaying the whole thing. The film is an intro you sit
# through once; after that the end card is the hub.
HUB_HREF = "/hero?at=end"

BACK_LINK = (
    f'<a class="back-link" href="{HUB_HREF}">&larr;&nbsp; Back to hub</a>'
)

GRADIO_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
<meta name="color-scheme" content="light">
<script>
// Gradio 6 picks its light/dark palette from the OS preference at runtime.
// The hub is designed on a light silver ground, so the light palette is
// forced and kept: a dark-mode machine would otherwise render Gradio's dark
// base underneath a light theme and leave half the controls unreadable.
(function () {
  function forceLight() {
    document.documentElement.classList.remove('dark');
    document.documentElement.style.colorScheme = 'light';
  }
  forceLight();
  document.addEventListener('DOMContentLoaded', forceLight);
  new MutationObserver(forceLight).observe(document.documentElement, {
    attributes: true, attributeFilter: ['class']
  });
})();

// A slim bar linking back to the landing page. Gradio rebuilds its DOM on
// interaction, so this re-attaches itself if it ever gets torn out.
(function () {
  function mount() {
    if (document.getElementById('hub-bar')) return;
    var host = document.querySelector('.gradio-container');
    if (!host) return;
    var bar = document.createElement('div');
    bar.id = 'hub-bar';
    bar.innerHTML = '<a href="__HUB_HREF__">\u2190 &nbsp;AI.PROJECTS.HUB</a>' +
                    '<span><span class="dot"></span>all inference local</span>';
    host.prepend(bar);
  }
  document.addEventListener('DOMContentLoaded', mount);
  new MutationObserver(mount).observe(document.documentElement, {childList: true, subtree: true});
  mount();
})();

// Live anonymizer in headless mode (see build_anon_browser_cam). While the
// output image is on screen: open the camera, then repeatedly grab a frame,
// POST it as a JPEG, and show the anonymized JPEG that comes back. The next
// frame is sent only after the previous answer arrives, so a slow model lowers
// the frame rate instead of piling up requests. Leaving the tab stops the
// camera, which also turns off the camera light.
(function () {
  var stream = null;     // the camera's MediaStream while it is open
  var starting = false;  // waiting for the permission prompt
  var blocked = false;   // permission refused; retry only after leaving the tab
  var lastUrl = null;    // object URL of the frame on screen, freed when replaced

  function byId(id) { return document.getElementById(id); }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  function setMsg(text) {
    var msg = byId('anon-live-msg');
    if (msg) msg.textContent = text;
  }

  // offsetParent is null when the element or a parent is display:none, which
  // is how Gradio hides inactive tabs and pages.
  function onScreen() {
    var out = byId('anon-live-out');
    return !!out && out.offsetParent !== null && !document.hidden;
  }

  function readControls() {
    var mode = document.querySelector('#anon-live-mode input:checked');
    var blur = document.querySelector('#anon-live-blur input[type=number]');
    return { mode: mode ? mode.value : 'basic', blur: blur ? blur.value : '51' };
  }

  function stopCamera() {
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
    stream = null;
  }

  async function startCamera() {
    starting = true;
    setMsg('Waiting for camera permission...');
    try {
      var s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (onScreen()) { stream = s; loop(s); }
      else s.getTracks().forEach(function (t) { t.stop(); });  // left while asking
    } catch (err) {
      blocked = true;
      setMsg('Camera unavailable: ' + err.message);
    }
    starting = false;
  }

  async function loop(s) {
    // The video element is never added to the page: it only decodes the
    // camera so frames can be copied onto the canvas.
    var video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.srcObject = s;
    await video.play();
    var canvas = document.createElement('canvas');
    setMsg('');

    while (stream === s) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      var frame = await new Promise(function (r) { canvas.toBlob(r, 'image/jpeg', 0.8); });

      var c = readControls();
      try {
        var res = await fetch('/api/anon/frame?mode=' + encodeURIComponent(c.mode) +
                              '&blur=' + encodeURIComponent(c.blur),
                              { method: 'POST', body: frame });
        if (!res.ok) throw new Error('server returned ' + res.status);
        var result = await res.blob();
        var out = byId('anon-live-out');
        if (out && stream === s) {
          if (lastUrl) URL.revokeObjectURL(lastUrl);
          lastUrl = URL.createObjectURL(result);
          out.src = lastUrl;
          setMsg('');
        }
      } catch (err) {
        setMsg('Server not responding (' + err.message + '), retrying...');
        await sleep(1000);
      }
    }
  }

  // Check twice a second whether the camera should be on or off.
  setInterval(function () {
    if (onScreen()) {
      if (!stream && !starting && !blocked) startCamera();
    } else {
      stopCamera();
      blocked = false;
    }
  }, 500);
})();
</script>
"""

# Gradio 6 moved theme and css from the Blocks constructor to launch(). Passing
# them to the wrong one is a deprecation warning on 6 and a TypeError on 5, so
# the version decides where they go.
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
STYLE_KWARGS = {"theme": gr.themes.Soft()}


BROWSER_CAM_NOTE = (
    "Click the webcam box and allow camera access, then press record to start "
    "streaming. Browsers only allow the camera on `localhost` or HTTPS, so "
    "open the hub at http://localhost:7860 rather than by IP address."
)


def build_anon_native_cam():
    """Live anonymizer controls that drive a native OpenCV window."""
    gr.Markdown("Opens a fast native window. Press `q` there to close it.")
    with gr.Row():
        with gr.Column():
            mode = gr.Radio(["basic", "perfect"], value="perfect", label="Anonymization mode")
            blur = gr.Slider(11, 151, value=51, step=2, label="Blur intensity")
        with gr.Column():
            btn_start = gr.Button("Start Live Feed", variant="primary", size="lg")
            btn_stop = gr.Button("Stop Live Feed")
            status = gr.Textbox(label="Status", interactive=False)

    mode.change(anon_update_mode, mode, status)
    blur.change(anon_update_blur, blur, status)
    btn_start.click(anon_start_cam, [mode, blur], status)
    btn_stop.click(anon_stop_cam, None, status)


def build_anon_browser_cam():
    """Live anonymizer fed by the browser's webcam, for headless mode.

    Only the anonymized output is shown. The script in GRADIO_HEAD finds these
    elements by id, starts the camera while this tab is visible, and reads the
    mode and blur controls on every frame, so they need no Gradio events.
    """
    gr.Markdown(
        "Allow camera access when the browser asks and the anonymized feed "
        "starts on its own. Browsers only allow the camera on `localhost` or "
        "HTTPS, so open the hub at http://localhost:7860 rather than by IP address."
    )
    gr.HTML(
        '<p id="anon-live-msg">Waiting for the camera...</p>'
        '<img id="anon-live-out" alt="Anonymized webcam feed" '
        'style="display:block; max-width:100%; max-height:460px; margin:0 auto;">'
    )
    with gr.Row():
        # Basic by default: perfect mode segments every face on the CPU here,
        # which is too slow to feel live on most machines. interactive=True is
        # required: Gradio renders a component with no events as read-only.
        gr.Radio(["basic", "perfect"], value="basic", label="Anonymization mode",
                 info="Perfect is much slower without a GPU.", elem_id="anon-live-mode",
                 interactive=True)
        gr.Slider(11, 151, value=51, step=2, label="Blur intensity", elem_id="anon-live-blur",
                  interactive=True)


def build_pose_native_cam():
    """Live rep counter controls that drive a native OpenCV window."""
    gr.Markdown("Opens a fast native window. Press `q` there to close it.")
    with gr.Row():
        with gr.Column():
            exercise = gr.Radio(EXERCISE_NAMES, value=EXERCISE_NAMES[0], label="Exercise")
            show_3d = gr.Checkbox(label="Show 3D skeleton viewer", value=False)
        with gr.Column():
            btn_start = gr.Button("Start Live Tracker", variant="primary", size="lg")
            btn_stop = gr.Button("Stop Tracker")
            status = gr.Textbox(label="Status", interactive=False)

    exercise.change(pose_update_exercise, exercise, status)
    show_3d.change(pose_update_3d, show_3d, status)
    btn_start.click(pose_start_cam, [exercise, show_3d], status)
    btn_stop.click(pose_stop_cam, None, status)


def build_pose_browser_cam():
    """Live rep counter fed by the browser's webcam, for headless mode."""
    gr.Markdown(BROWSER_CAM_NOTE)
    session = gr.State(None, delete_callback=close_pose_session)
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, type="numpy",
                       label="Your webcam", height=460)
        out = gr.Image(label="Rep counter", height=460)
    with gr.Row():
        exercise = gr.Radio(EXERCISE_NAMES, value=EXERCISE_NAMES[0], label="Exercise")
        with gr.Column():
            show_3d = gr.Checkbox(label="Show 3D skeleton viewer", value=False)
            btn_reset = gr.Button("Reset Counter")
            status = gr.Markdown()

    cam.stream(pose_stream_frame, [cam, exercise, show_3d, session], [out, session],
               stream_every=STREAM_EVERY)
    btn_reset.click(pose_stream_reset, session, status)


def build_ui():
    blocks_style = {} if GRADIO_MAJOR >= 6 else STYLE_KWARGS
    with gr.Blocks(title="AI Projects Hub", **blocks_style) as app:

        def show_page(name):
            names = ["landing", "anon", "depth", "pose", "search"]
            return tuple(gr.update(visible=name == n) for n in names)

        # ---------------------------------------------------------- landing
        with gr.Group(visible=True) as page_landing:
            gr.Markdown(
                "# AI Projects Hub\n"
                "Four computer-vision and NLP mini-projects, each running fully "
                "on your own machine. Nothing is uploaded anywhere. Pick one to begin."
            )
            with gr.Row():
                btn_anon = gr.Button("Face Anonymizer\nOcclusion-aware blurring",
                                     size="lg", variant="primary", elem_classes="hub-card")
                btn_depth = gr.Button("Depth Map Visualizer\n2D photos into 3D geometry",
                                      size="lg", variant="primary", elem_classes="hub-card")
            with gr.Row():
                btn_pose = gr.Button("AI Rep Counter\nPose tracking and rep counting",
                                     size="lg", variant="primary", elem_classes="hub-card")
                btn_search = gr.Button("Semantic Search\nSearch the docs by meaning",
                                       size="lg", variant="primary", elem_classes="hub-card")
            gr.Markdown(
                "<sub>Models load the first time each project is opened, so the "
                "first run of a project is slower than the ones after it.</sub>"
            )

        # ------------------------------------------------------- 1. faces
        with gr.Group(visible=False) as page_anon:
            gr.HTML(BACK_LINK)
            gr.Markdown(
                "## Face Anonymizer\n"
                "**Basic** blurs the whole detection box. **Perfect** segments the face "
                "itself, so a hand or microphone held in front of it stays sharp."
            )

            with gr.Tabs():
                with gr.Tab("Upload Image/Video"):
                    with gr.Row():
                        with gr.Column():
                            anon_input = gr.File(label="Image or video", height=260)
                            anon_mode = gr.Radio(["basic", "perfect"], value="perfect",
                                                 label="Anonymization mode")
                            anon_blur = gr.Slider(11, 151, value=51, step=2,
                                                  label="Blur intensity")
                            anon_btn = gr.Button("Process File", variant="primary")
                        with gr.Column():
                            anon_out_img = gr.Image(label="Output image", visible=True, height=460)
                            anon_out_vid = gr.Video(label="Output video", visible=False, height=460)
                            anon_status = gr.Markdown()

                with gr.Tab("Live Webcam"):
                    if HEADLESS:
                        build_anon_browser_cam()
                    else:
                        build_anon_native_cam()

            def update_anon_vis(file):
                if not file:
                    return gr.update(visible=True), gr.update(visible=False)
                video = is_video(file)
                return gr.update(visible=not video), gr.update(visible=video)

            anon_input.change(update_anon_vis, anon_input, [anon_out_img, anon_out_vid])
            anon_btn.click(anon_process_media_ui,
                           [anon_input, anon_mode, anon_blur],
                           [anon_out_img, anon_out_vid, anon_status])

        # ------------------------------------------------------- 2. depth
        with gr.Group(visible=False) as page_depth:
            gr.HTML(BACK_LINK)
            gr.Markdown(
                "## Depth Map Visualizer\n"
                "Estimates relative depth from a single image. Depth order is reliable; "
                "the values are not real-world distances."
            )

            with gr.Row():
                with gr.Column():
                    depth_input = gr.File(label="Image(s) or video(s)",
                                          file_types=["image", "video"],
                                          file_count="multiple", height=260)
                    depth_model = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[0],
                                              label="Depth estimation model")
                    depth_cmap = gr.Dropdown(COLORMAP_CHOICES, value="inferno", label="Colormap")
                    with gr.Row():
                        depth_inv = gr.Checkbox(label="Invert depth", value=False)
                        depth_sbs = gr.Checkbox(label="Side-by-side", value=True)
                        depth_leg = gr.Checkbox(label="Colorbar legend", value=False)
                    with gr.Row():
                        depth_ply = gr.Checkbox(label="Point cloud (.ply)", value=False)
                        depth_npy = gr.Checkbox(label="Raw depth (.npy)", value=False)
                        depth_mesh = gr.Checkbox(label="3D mesh (images only)", value=False)
                    depth_btn = gr.Button("Generate Depth Maps", variant="primary")

                with gr.Column():
                    with gr.Tabs():
                        with gr.Tab("Images & Videos"):
                            depth_out_gal = gr.Gallery(label="Preview", columns=2, height=520)
                        with gr.Tab("3D Viewer"):
                            depth_out_3d = gr.Model3D(label="Interactive depth mesh",
                                                      height=520)
                    depth_out_files = gr.File(label="All outputs (downloads)",
                                              file_count="multiple",
                                              interactive=False, height=200)
                    depth_status = gr.Markdown()

            depth_btn.click(
                depth_process_media_ui,
                [depth_input, depth_model, depth_cmap, depth_inv, depth_sbs,
                 depth_leg, depth_ply, depth_npy, depth_mesh],
                [depth_out_gal, depth_out_files, depth_out_3d, depth_status],
            )

        # -------------------------------------------------------- 3. pose
        with gr.Group(visible=False) as page_pose:
            gr.HTML(BACK_LINK)
            gr.Markdown(
                "## AI Rep Counter\n"
                "Tracks 33 body landmarks, measures the joint angle that defines each "
                "exercise, and counts a rep once the movement passes through both the "
                "flexed and extended positions. Camera view is detected automatically."
            )

            with gr.Tabs():
                with gr.Tab("Live Webcam (recommended)"):
                    if HEADLESS:
                        build_pose_browser_cam()
                    else:
                        build_pose_native_cam()

                with gr.Tab("Upload Video"):
                    with gr.Row():
                        with gr.Column():
                            pose_input = gr.File(label="Video", file_types=["video"], height=260)
                            pose_ex_file = gr.Radio(EXERCISE_NAMES, value=EXERCISE_NAMES[0],
                                                    label="Exercise")
                            pose_3d_file = gr.Checkbox(label="Show 3D skeleton viewer",
                                                       value=False)
                            pose_btn = gr.Button("Process Video", variant="primary")
                        with gr.Column():
                            pose_out_vid = gr.Video(label="Output with rep counter", height=520)
                            pose_status = gr.Markdown()

                    pose_btn.click(pose_process_media_ui,
                                   [pose_input, pose_ex_file, pose_3d_file],
                                   [pose_out_vid, pose_status])

        # ------------------------------------------------------ 4. search
        with gr.Group(visible=False) as page_search:
            gr.HTML(BACK_LINK)
            gr.Markdown("## Semantic Document Search")

            with gr.Tabs():
                with gr.Tab("Search"):
                    gr.Markdown(
                        "Searches this hub's documentation, plus anything in your "
                        "library, by *meaning* rather than exact words, using "
                        "`all-MiniLM-L6-v2` embeddings in a FAISS index. "
                        "**Keyword** is a TF-IDF baseline for comparison; "
                        "**Hybrid** fuses both."
                    )

                    with gr.Row():
                        with gr.Column(scale=3):
                            search_query = gr.Textbox(
                                label="Query",
                                placeholder="e.g. 'How do I hide faces?' or 'calculating 3d joints'",
                            )
                        with gr.Column(scale=1):
                            search_method = gr.Radio(SEARCH_METHODS, value="Semantic",
                                                     label="Method")
                            search_topk = gr.Slider(1, 10, value=3, step=1, label="Results")

                    search_btn = gr.Button("Search Documents", variant="primary")

                    gr.Markdown("**Try:** *how do I blur a video* · *what counts a squat* · "
                                "*exporting 3d point clouds* · *is any of this sent to the cloud*")
                    search_results = gr.Markdown("Results will appear here.")

                    with gr.Accordion("What is indexed?", open=False):
                        corpus_info = gr.Markdown("Click below to build the index.")
                        corpus_btn = gr.Button("Show indexed documents")
                        corpus_btn.click(search_corpus_summary, None, corpus_info)

                    search_inputs = [search_query, search_method, search_topk]
                    search_btn.click(perform_search, search_inputs, search_results)
                    search_query.submit(perform_search, search_inputs, search_results)

                with gr.Tab("Document Library"):
                    gr.Markdown(
                        "### Drop anything in here\n"
                        "Upload a pile of mixed files and have them sorted for you, "
                        "either by **what they are about** or by **what they are**. "
                        "Text is pulled out of PDFs, Word, Excel, PowerPoint, HTML, "
                        "ePub, RTF, notebooks, code and plain text; formats with no "
                        "readable text are filed by name and type. Everything added "
                        "here also becomes searchable in the Search tab."
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            lib_upload = gr.File(
                                label="Drop files here (any number, any mix of types)",
                                file_count="multiple", height=300,
                            )
                            with gr.Row():
                                lib_add_btn = gr.Button("Add to Library", variant="primary")
                                lib_clear_btn = gr.Button("Clear Library")
                            lib_status = gr.Markdown()
                        with gr.Column(scale=1):
                            lib_stats = gr.Markdown("*Library not loaded yet.*")
                            lib_mode = gr.Radio(ORGANIZE_MODES, value=ORGANIZE_MODES[0],
                                                label="Sort by")
                            lib_breadth = gr.Slider(
                                0.55, 0.95, value=DEFAULT_CLUSTER_DISTANCE, step=0.05,
                                label="Grouping breadth",
                                info="Lower = more, tighter topics. Higher = fewer, broader ones.",
                            )
                            lib_organize_btn = gr.Button("Organize", variant="primary")

                    lib_groups = gr.Markdown(
                        "Add some files, then press **Organize**."
                    )

                    with gr.Accordion("Supported file types", open=False):
                        gr.Markdown(
                            "Text is extracted from: "
                            + ", ".join(f"`{e}`" for e in supported_extensions())
                            + ".\n\nAnything else is still accepted and filed by "
                            "name and type."
                        )

                    # The breadth slider only affects content clustering.
                    lib_mode.change(
                        lambda m: gr.update(visible=m == ORGANIZE_MODES[0]),
                        lib_mode, lib_breadth,
                    )

                    lib_add_btn.click(library_add, [lib_upload, lib_mode, lib_breadth],
                                      [lib_status, lib_groups, lib_stats])
                    lib_organize_btn.click(library_organize, [lib_mode, lib_breadth],
                                           lib_groups)
                    lib_mode.change(library_organize, [lib_mode, lib_breadth], lib_groups)
                    lib_clear_btn.click(library_clear, [lib_mode, lib_breadth],
                                        [lib_status, lib_groups, lib_stats])

        # ------------------------------------------------------ navigation
        pages = [page_landing, page_anon, page_depth, page_pose, page_search]
        for button, target in (
            (btn_anon, "anon"), (btn_depth, "depth"),
            (btn_pose, "pose"), (btn_search, "search"),
        ):
            button.click(lambda t=target: show_page(t), None, pages)
        # Deep links, so the landing page can send you straight into a project
        # rather than dropping everyone on the hub index: /app?p=depth
        PROJECT_ALIASES = {
            "anon": "anon", "face": "anon", "anonymizer": "anon",
            "depth": "depth",
            "pose": "pose", "reps": "pose",
            "search": "search", "semantic": "search",
        }

        def open_from_url(request: gr.Request):
            try:
                wanted = (request.query_params.get("p") or "").strip().lower()
            except Exception:
                wanted = ""
            return show_page(PROJECT_ALIASES.get(wanted, "landing"))

        app.load(open_from_url, None, pages)

    return app


# ------------------------------------------------- main-thread webcam loops

def run_anon_webcam():
    """Own the camera until the UI clears the flag or the user closes the window."""
    cap = open_camera()
    if cap is None:
        print("Error: could not open the webcam.")
        hub_state['anon_running'] = False
        return

    cv2.namedWindow(ANON_WINDOW, cv2.WINDOW_NORMAL)
    try:
        while hub_state['anon_running']:
            ret, frame = cap.read()
            if not ret:
                break

            # Re-read the controls each frame so UI changes take effect live,
            # and segment at a lower resolution than file processing uses so
            # the window stays responsive.
            model = configured_anonymizer(hub_state['anon_mode'], hub_state['anon_blur'],
                                          parse_size=LIVE_PARSE_SIZE)

            start = time.time()
            output = model.process_frame(frame)
            fps = 1.0 / max(time.time() - start, 1e-6)

            cv2.putText(output,
                        f"{hub_state['anon_mode']} | blur {model.blur_intensity} | "
                        f"{fps:.0f} FPS | press 'q' to close",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(ANON_WINDOW, output)

            if pump_gui() in QUIT_KEYS or window_is_closed(ANON_WINDOW):
                break
    finally:
        hub_state['anon_running'] = False
        cap.release()
        close_window(ANON_WINDOW)


def run_pose_webcam():
    """Own the camera until the UI clears the flag or the user closes the window."""
    cap = open_camera()
    if cap is None:
        print("Error: could not open the webcam.")
        hub_state['pose_running'] = False
        return

    cv2.namedWindow(POSE_WINDOW, cv2.WINDOW_NORMAL)

    exercise, show_3d = hub_state['pose_exercise'], hub_state['pose_3d']
    processor = PoseProcessor(exercise=exercise, show_3d=show_3d)
    fps = None
    try:
        while hub_state['pose_running']:
            ret, frame = cap.read()
            if not ret:
                break

            # Changing exercise mid-stream rebuilds the processor, which also
            # resets the rep count: the old count belongs to the old exercise.
            if exercise != hub_state['pose_exercise'] or show_3d != hub_state['pose_3d']:
                exercise, show_3d = hub_state['pose_exercise'], hub_state['pose_3d']
                processor.close()
                processor = PoseProcessor(exercise=exercise, show_3d=show_3d)

            start = time.time()
            output = processor.process_frame(frame, fps=fps)
            cv2.imshow(POSE_WINDOW, output)
            # Measured after the fact, so the overlay shows the previous
            # frame's rate rather than blocking to compute the current one.
            fps = 1.0 / max(time.time() - start, 1e-6)

            if pump_gui() in QUIT_KEYS or window_is_closed(POSE_WINDOW):
                break
    finally:
        hub_state['pose_running'] = False
        processor.close()
        cap.release()
        close_window(POSE_WINDOW)


def build_web_app(blocks):
    """Serve the landing page at / and mount the Gradio app at /app."""
    web = FastAPI(title="AI Projects Hub")

    web.mount("/static", StaticFiles(directory=str(WEB_DIR / "assets")), name="static")

    @web.middleware("http")
    async def no_stale_assets(request, call_next):
        """Make the browser revalidate front-end assets on every request.

        StaticFiles sends only an etag, so browsers are free to heuristically
        cache an ES module and keep running an old build after an edit. That
        is invisible and deeply confusing while iterating on the design.
        """
        response = await call_next(request)
        # The HTML pages too: the film's tuning numbers live inline in them.
        pages = ("/", "/hero", "/panel", "/lab")
        if request.url.path.startswith("/static/") or request.url.path in pages:
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    @web.get("/")
    def landing():
        """The scroll film. This is the landing page."""
        return FileResponse(WEB_DIR / "hero.html")

    @web.get("/hero")
    def hero():
        """Kept so older links to the scroll film still resolve."""
        return FileResponse(WEB_DIR / "hero.html")

    @web.get("/panel")
    def panel():
        """The earlier instrument-panel landing page."""
        return FileResponse(WEB_DIR / "index.html")

    @web.get("/lab")
    def lab():
        """The Resolve tuning lab: the effect plus a live parameter panel."""
        return FileResponse(WEB_DIR / "lab.html")

    @web.post("/api/anon/frame")
    async def anon_frame(request: Request, mode: str = "basic", blur: int = 51):
        """One frame of the browser-webcam anonymizer: JPEG in, JPEG out."""
        if mode not in ("basic", "perfect"):
            return Response("mode must be basic or perfect", status_code=400)
        blur = min(max(blur, 11), 151)
        jpeg = await request.body()
        # The model call blocks for tens of milliseconds or more, so it runs
        # on a worker thread rather than stalling the server's event loop.
        result = await run_in_threadpool(anon_browser_frame, jpeg, mode, blur)
        if result is None:
            return Response("body is not a readable image", status_code=400)
        return Response(result, media_type="image/jpeg")

    @web.get("/api/status")
    def status():
        """Real telemetry for the landing page's readouts.

        Everything here is read from live process state. Nothing is invented:
        an instrument panel that reports numbers it did not measure is worse
        than no panel at all.
        """
        import torch

        if torch.cuda.is_available():
            device, detail = "cuda", torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            device, detail = "mps", "Apple Silicon GPU"
        else:
            device, detail = "cpu", "no accelerator detected"

        # Report the models that are actually resident, not the ones that exist.
        friendly = {
            "anon_basic": "face:basic", "anon_perfect": "face:perfect",
            "embedder": "minilm", "search": "faiss", "library": "library",
        }
        names = [friendly.get(k, k.replace("depth_", "depth:").split("/")[-1])
                 for k in _models if k != "library"]

        engine = _models.get("search")
        library = _models.get("library")

        return JSONResponse({
            "device": device,
            "device_detail": detail,
            "uptime": int(time.time() - STARTED_AT),
            "models": {"loaded": len(names), "names": names},
            "search": {
                "built": engine is not None,
                "chunks": len(engine.documents) if engine else 0,
                "documents": len({m["file"] for m in engine.doc_metadata}) if engine else 0,
            },
            "library": {
                "files": len(library.files) if library else 0,
                "categories": len({f.category for f in library.files}) if library else 0,
                "readable": sum(1 for f in library.files if f.status == "ok") if library else 0,
            },
            "camera": {
                "anonymizer": hub_state["anon_running"],
                "pose": hub_state["pose_running"],
            },
            "events": list(EVENTS),
        })

    # Gradio 6 moved theme, css and head from the Blocks constructor onto
    # mount_gradio_app. Filtering against the real signature keeps this
    # working across versions instead of guessing which release moved what.
    import inspect
    supported = inspect.signature(gr.mount_gradio_app).parameters
    styling = {
        "theme": gr.themes.Soft(),
        "css_paths": [str(WEB_DIR / "assets" / "gradio-theme.css")],
        "head": GRADIO_HEAD.replace("__HUB_HREF__", HUB_HREF),
    }
    styling = {k: v for k, v in styling.items() if k in supported}
    if len(styling) < 3:
        missing = {"theme", "css_paths", "head"} - set(styling)
        print(f"Note: this Gradio version ignores {', '.join(sorted(missing))}; "
              "the app will render unstyled.")

    return gr.mount_gradio_app(web, blocks, path="/app", **styling)


def main():
    app = build_ui()
    web = build_web_app(app)

    # uvicorn runs on a worker thread so the main thread stays free for the
    # OpenCV windows, which macOS requires be driven from the main thread.
    config = uvicorn.Config(web, host=HUB_HOST, port=HUB_PORT, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

    while not server.started:
        time.sleep(0.05)

    url = f"http://{'127.0.0.1' if HUB_HOST in ('0.0.0.0', '::') else HUB_HOST}:{HUB_PORT}"
    log_event(f"hub online at {url}", "ok")
    print("\n  AI PROJECTS HUB")
    print(f"  landing  {url}")
    print(f"  app      {url}/app")
    if HEADLESS:
        print("  headless mode: live webcam streams through the browser.")
    print("  Ctrl+C to quit.\n")

    if OPEN_BROWSER:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass

    if HEADLESS:
        # No display and no camera to drive, so the main thread just waits for
        # the server thread instead of pumping an OpenCV event loop.
        try:
            while not server.should_exit:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nShutting down the hub.")
        finally:
            server.should_exit = True
        return

    try:
        while True:
            if hub_state['anon_running']:
                log_event("camera opened: anonymizer")
                run_anon_webcam()
                log_event("camera released: anonymizer")
            elif hub_state['pose_running']:
                log_event("camera opened: rep counter")
                run_pose_webcam()
                log_event("camera released: rep counter")
            else:
                # Keep draining the event queue while idle. If a window is
                # somehow still on screen, this is what lets it finish closing
                # instead of sitting there beachballing.
                cv2.waitKey(1)
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nShutting down the hub.")
    finally:
        close_window(ANON_WINDOW)
        close_window(POSE_WINDOW)
        server.should_exit = True


if __name__ == "__main__":
    main()
