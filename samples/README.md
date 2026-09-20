# Sample output

Real output from the four projects, kept small so the work can be looked at
without installing anything. Every file here was produced by the code in this
repository; nothing was edited by hand.

| File | Project | What it shows |
|---|---|---|
| `depth-side-by-side.png` | Depth Map Visualizer | The source photo beside its estimated depth map, inferno colormap, with the relative-depth scale. Near is dark, far is bright. |
| `depth-map.png` | Depth Map Visualizer | A depth map on its own, from a second image. |
| `depth-mesh.ply` | Depth Map Visualizer | The same estimate back-projected into a 3D mesh. Opens in MeshLab, Blender, or macOS Preview. |
| `rep-counter.mp4` | AI Rep Counter | Nine seconds of bicep curls. The left panel tracks the skeleton and elbow angle; the right is the 3D landmark view. Watch the RIGHT counter go from 1 to 2. |

## A note on what is missing

There is no face-anonymizer sample here. The project works — it is exercised in
the test run described in the main README — but no photograph containing a face
is checked into this repository, and producing a sample from the rep-counter
footage would have meant blurring a frame that already has pose landmarks drawn
across the face, which would misrepresent what the anonymizer actually outputs.

To generate one, drop any photo with a face into the project root and run:

```bash
python face_anonymizer/main.py --input yourphoto.jpg --mode perfect
```

The result lands in `output/anonymizer/`.

## Reproducing these

```bash
docker compose up --build        # then open http://localhost:7860
```

Or without Docker, per the main README. The depth samples came from the Depth
Map Visualizer tab, the video from the AI Rep Counter tab.

The gym footage is a frame of a publicly posted exercise tutorial, used here
only as test input for coursework.
