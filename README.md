# 🚀 AI Projects Hub

Four computer-vision and NLP mini-projects for the Applied AI Programming
course, behind a single Gradio interface. Every model runs **locally** — no
image, video, or document ever leaves the machine.

| | Project | What it does | Core technology |
|---|---|---|---|
| 🕵️ | [Face Anonymizer](face_anonymizer/README.md) | Blurs faces while leaving objects in front of them sharp | MediaPipe BlazeFace + Segformer face parsing |
| 🌌 | [Depth Map Visualizer](Depth%20Map%20Visualization/README.md) | Estimates depth from a single photo; exports 3D point clouds and meshes | Depth Anything V2 / Intel DPT |
| 🏋️ | [AI Rep Counter](pose_estimator/README.md) | Tracks 33 body landmarks and counts exercise repetitions | MediaPipe Pose Landmarker |
| 🔎 | [Semantic Search](semantic_search/README.md) | Searches docs by meaning; drop-anything file library that sorts by content | all-MiniLM-L6-v2 + FAISS |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python hub.py
```

The hub opens at **http://127.0.0.1:7860**.

## Docker

```bash
docker compose up --build
```

Same hub, same address: **http://localhost:7860**. The build installs the
CPU-only build of PyTorch, so the image stays around 3 GB rather than the 8 GB
the CUDA wheels would cost.

Three things are mounted rather than kept in the image, so nothing is lost when
the container is rebuilt:

| Mount | Holds |
|---|---|
| `./output` | Everything the four projects write |
| `./semantic_search/data` | The document library and its search index |
| `model-cache` volume | Model weights, downloaded on first use |

**The live webcam is off in the container.** It draws native OpenCV windows and
opens `/dev/video0`, neither of which exists inside one; the buttons say so if
you press them. Everything else — image and video upload, depth export, the
document library, search — works exactly as it does locally. Run `python hub.py`
on the host when you want the camera.

The container reads four environment variables, so the same image also runs
outside Compose:

| Variable | Default | |
|---|---|---|
| `HUB_HOST` | `127.0.0.1` | `0.0.0.0` in the image, so the port maps out |
| `HUB_PORT` | `7860` | |
| `HUB_HEADLESS` | off | `1` disables the webcam windows |
| `HUB_OPEN_BROWSER` | on | `0` stops it launching a browser on startup |

```bash
docker run -p 7860:7860 -v "$PWD/output:/app/output" ai-projects-hub
```

| | |
|---|---|
| `/` | Landing page — the scroll film |
| `/app` | The working Gradio interface |
| `/panel` | The earlier instrument-panel landing page |
| `/lab` | Resolve tuning lab — the effect plus a live parameter panel |
| `/api/status` | JSON telemetry the landing page polls |

Models are downloaded on first use and cached in `~/.cache/huggingface`, so the
first run of each project is slow and every run after it is fast. Nothing is
loaded until you actually open a project, which keeps startup to a second or two.

## Running a project on its own

Each project also works standalone:

```bash
python "Depth Map Visualization/app.py"                       # its own web UI, port 7861
python "Depth Map Visualization/depth_visualizer.py" -i photo.jpg --sbs --ply
python face_anonymizer/main.py --input video.mp4 --mode perfect
python semantic_search/search_engine.py --method hybrid "how do I blur a face"
```

## Document library

The Semantic Search project has a **Document Library** tab: drop in any pile of
mixed files and it sorts them either by **what they are about** (clustering the
content, ignoring format) or by **what they are** (file type). Text is pulled
out of PDFs, Word, Excel, PowerPoint, HTML, ePub, RTF, notebooks and code, and
everything added becomes searchable. Files with no readable text are still
accepted and filed by name and type.

## Live webcam

Two projects offer a live camera feed. Clicking **Start Live Feed** opens a
native OpenCV window rather than streaming into the browser — macOS requires
all GUI windows to be driven from the main thread, so the buttons in the web UI
set a flag that the main loop in `hub.py` picks up and acts on. It is also
simply much faster than round-tripping frames through the browser.

Press `q` (or `Esc`), click the window's close button, or press **Stop** to end
it. Only one project can hold the camera at a time; the UI will say so if you
try to start both.

## Sample output

`samples/` holds real output from the projects — depth maps, a 3D mesh, and a
clip of the rep counter — so the work can be looked at without installing
anything. `samples/README.md` says what each file is.

## Where output goes

```
output/
├── anonymizer/   blurred images and video
├── depth/        depth maps, point clouds, meshes, raw .npy arrays
└── pose/         annotated video with the rep counter
```

Filenames are timestamped, so a new run never overwrites an earlier result.

## The interface

The landing page at `/` is hand-built HTML, CSS and canvas; the Gradio app is
mounted underneath at `/app` and restyled to match. A FastAPI process serves
both, with uvicorn on a worker thread so the main thread stays free for the
OpenCV windows that macOS insists be driven from it.

Its readouts are **real**. `/api/status` reports the compute device actually
detected, which models are actually resident, the real size of the search
index and library, and a log of events that actually happened. An instrument
panel that reports numbers it did not measure is worse than no panel.

The hero is a height field rendered as contour lines — the same back-projection
the depth project performs, running live. It stops drawing when scrolled out of
view or when the tab is hidden, and holds still entirely under
`prefers-reduced-motion`.

## Layout

```
.
├── hub.py                      # Gradio app, FastAPI server, webcam main loop
├── requirements.txt            # every dependency for all four projects
├── Dockerfile                  # CPU-only image for the whole hub
├── docker-compose.yml          # ports, volumes, headless environment
├── web/
│   ├── hero.html               # the landing page
│   ├── index.html              # the earlier instrument panel, at /panel
│   └── assets/
│       ├── tokens.css          # design tokens
│       ├── landing.css         # landing page styles
│       ├── landing.js          # canvas field, reveals, telemetry polling
│       └── gradio-theme.css    # restyles /app to match
├── face_anonymizer/
│   ├── main.py                 # FaceAnonymizer class + CLI
│   └── blaze_face_short_range.tflite
├── Depth Map Visualization/
│   ├── depth_visualizer.py     # estimation, colour mapping, 3D export + CLI
│   └── app.py                  # standalone web UI
├── pose_estimator/
│   ├── pose_counter.py         # EXERCISES table, RepCounter, PoseProcessor
│   └── pose_landmarker_full.task
└── semantic_search/
    ├── search_engine.py        # chunking, FAISS + TF-IDF indexes, CLI
    ├── file_library.py         # text extraction, clustering, type grouping
    └── data/
        └── library/            # files added through the Document Library tab
```

## Notes on responsible use

Each project's README has a full report section covering its limitations,
privacy implications, and failure modes. Three points are worth repeating here:

- **The anonymizer can miss a face.** A single undetected frame is a privacy
  breach. Review processed video before releasing it.
- **Depth is relative, not metric.** The values order objects correctly but are
  not distances. Never use them where a real measurement is required.
- **The rep counter is not a coach.** It measures one joint angle. It cannot
  see form, load, or pain, and a high count is not evidence of safe technique.
