# Project 3: Exercise Rep Counter with Pose Estimation

A mini-application that uses computer vision to count exercise repetitions from
video files or a live webcam. This project fulfils the "Pose Estimation"
learning goals and practical requirements for the Applied AI Programming course.

## Overview

The application uses the **MediaPipe Pose Landmarker** to detect 33 human body
key points in real time. By connecting three specific landmarks it calculates a
joint angle (elbow, knee, or shoulder) and runs a **two-state machine** over
that angle to count repetitions.

## Features
- **Live and video support:** processes pre-recorded video files and live
  webcam streams through the Gradio web UI.
- **Five exercises:** Bicep Curl, Squat, Push-up, Lunge, and Jumping Jack.
- **Independent bilateral tracking:** counts the left and right sides
  separately for asymmetric exercises such as alternating bicep curls, and
  reports a single shared total for symmetric ones such as squats.
- **Movement smoothing:** an Exponential Moving Average over the angle
  suppresses landmark jitter that would otherwise register phantom reps.
- **Missing landmark handling:** joints below the model's 0.5 visibility
  threshold pause counting for that side rather than producing garbage.
- **Automatic view detection:** reports whether the camera sees the athlete
  from the front or the side, based on the projected shoulder separation.
- **Optional 3D skeleton viewer:** renders the pose world landmarks in 3D
  beside the video.
- **Visual feedback:** draws the skeleton and an on-screen display of the
  current angles and rep counts directly onto the video.

## Adding an exercise

Every exercise is one entry in the `EXERCISES` table in `pose_counter.py`. No
other code changes are needed:

```python
"Shoulder Press": ExerciseSpec(
    joint="Elbow",
    left=(L_SHOULDER, L_ELBOW, L_WRIST),
    right=(R_SHOULDER, R_ELBOW, R_WRIST),
    flexed_below=70, extended_above=160, symmetric=True,
),
```

## Report & Answers to Evaluation Criteria

### 1. What pose estimation model was used?
The **MediaPipe Pose Landmarker**, accessed through the modern
`mediapipe.tasks.python.vision` Tasks API rather than the older
`mediapipe.solutions.pose` interface. The bundled `pose_landmarker_full.task`
model is used, with detection, presence, and tracking confidences all set to
`0.5`. The Tasks API was chosen because it exposes `pose_world_landmarks`,
metric 3D coordinates in metres relative to the hips, which is what makes the
optional 3D skeleton viewer possible.

### 2. Which exercises were counted and which angles were used?
| Exercise | Joint angle | Landmarks |
|---|---|---|
| Bicep Curl | Elbow | Shoulder → Elbow → Wrist |
| Squat | Knee | Hip → Knee → Ankle |
| Push-up | Elbow | Shoulder → Elbow → Wrist |
| Lunge | Knee | Hip → Knee → Ankle |
| Jumping Jack | Shoulder (armpit) | Hip → Shoulder → Wrist |

The angle is computed with `arctan2` on the 2D normalised image coordinates of
the three landmarks, taking the middle one as the vertex.

### 3. What movement phases were detected and how was a repetition defined?
Each exercise moves a joint between two poses: **extended** (a large angle) and
**flexed** (a small angle). Every exercise declares both thresholds, and one
repetition is counted each time the athlete returns to the extended pose
*having already reached full flexion*.

Requiring both thresholds to be crossed is what makes the count trustworthy.
Angles in the band between them leave the phase untouched, so a partial
movement that never reaches the flexed position is correctly ignored rather
than counted. Bouncing gently at the top of a squat, for example, registers
nothing at all.

- **Bicep Curl:** extended above 150°, flexed below 50°.
- **Squat and Lunge:** extended above 160°, flexed below 100°.
- **Push-up:** extended above 150°, flexed below 90°.
- **Jumping Jack:** arms overhead is extended (above 120°), arms at the sides
  is flexed (below 45°).

Three more rules sit around those thresholds:

- **Smoothing.** Each raw angle is blended into a running average,
  `smoothed = 0.3 × new + 0.7 × previous`, before it is compared with the
  thresholds. A single jittery frame therefore cannot cross a threshold on its
  own.
- **Visibility.** If any of the three landmarks for a side has a visibility
  score below `0.5`, that side is skipped for the frame: its phase and count
  are frozen rather than updated from a guessed position.
- **Left, right, and total.** Each side keeps its own phase and count. For
  exercises that move one side at a time (Bicep Curl, Lunge) the total is
  left + right. For exercises where both sides move together (Squat,
  Push-up, Jumping Jack) the total is the larger of the two, so one squat is
  not counted twice.

The on-screen **Front / Side** label comes from the horizontal gap between
the two shoulders: wider than 12% of the frame width is reported as Front,
otherwise Side.

### 4. What kinds of videos work well vs. are difficult?
- **Works well:** footage shot side-on or at a slight angle, well lit, with the
  tracked joints unoccluded and the whole limb in frame. Controlled,
  deliberate movements are counted essentially perfectly.
- **Difficult:**
  - **Front-facing squats.** The knee angle is measured from 2D projected
    coordinates, so facing the camera head-on can leave the projected angle
    almost unchanged through a deep squat. The on-screen view detector warns
    when the camera is front-on for this reason.
  - **Baggy clothing**, which obscures joints and drops the model's visibility
    score below the counting threshold.
  - **Fast or jerky movement**, where motion blur causes missed detections.
  - **Partial reps**, which are deliberately not counted. This is correct
    behaviour, but it can look like a missed count to a user who is not going
    through the full range of motion.

### 5. Privacy, Safety, and Responsible Use
**Privacy.** Exercise footage reveals a great deal: body shape, home interiors,
and sometimes health conditions. MediaPipe runs entirely locally in Python
here, so no video data ever leaves the machine.

**Safety and overtrust.** This counter is built from simple geometric
heuristics on a handful of landmarks. It is *not* a substitute for medical or
physiotherapy advice. It does not assess spinal alignment or judge whether form
is safe, and it cannot see load, fatigue, or pain. Treating a high rep count as
evidence of good technique would be actively dangerous. It is an analytical
programming exercise, not a coaching tool.

**Bias.** Pose models are trained on datasets that under-represent some body
types, skin tones, mobility aids, and adaptive athletes. Accuracy is not
uniform across users, and a lower rep count may say more about the model's
training data than about the person exercising.
