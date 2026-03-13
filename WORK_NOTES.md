# Multi-Camera Calibration & Point Cloud Fusion — Work Notes

## Overview

This project calibrates a 5-camera RGBD array and fuses their depth streams into a
single point cloud in Camera 1's coordinate frame. The array consists of:

| Camera | Model          | Role      | Serial         |
|--------|----------------|-----------|----------------|
| cam1   | Intel L515     | Reference | f0210390       |
| cam2   | Intel L515     | —         | f0171709       |
| cam3   | ZED-M (left)   | —         | N/A (V4L2)     |
| cam4   | Intel D405     | —         | 130322271201   |
| cam5   | Intel D405     | —         | 218622270381   |
| t265   | Intel T265     | Tracking  | 905312111793   |

Camera 1 is the reference frame. All other cameras are expressed relative to it.
The T265 provides 6DOF pose tracking but is not extrinsically calibrated to the array.

---

## System Environment

- **OS:** Ubuntu, Wayland display (DISPLAY=:0, XWayland running)
- **GPU:** AMD (no CUDA, no ROCm)
- **OpenCV:** 4.6.0 — uses the **old ArUco API** (`cv::Ptr<CharucoBoard>`, `detectMarkers` +
  `interpolateCornersCharuco`). No `CharucoDetector`, no new `CharucoBoard` constructor.
- **librealsense2:** installed at `/usr/local/lib/cmake/realsense2/`
- **Open3D:** 0.19.0, installed at `/usr/local/lib/cmake/Open3D/`
- **ZED SDK:** NOT installed. ZED camera is accessed via V4L2/OpenCV VideoCapture on
  `/dev/video12` (set as `dev: 12` in `configs/cameras.yaml`).

---

## Project Layout

```
multi_cam_calib/
├── CMakeLists.txt
├── calibrate_filtered.py               # Python stereo calib with iterative outlier removal
├── calibrate_relay.py                  # relay calibration (independent solvePnP per camera)
├── optimize_poses.py                   # pose graph optimization over all edges
├── configs/
│   └── cameras.yaml                    # camera models, serials, V4L2 device index
├── include/
│   └── calib_common.hpp                # shared helpers: board, ArUco detect, transforms
├── src/
│   ├── capture_pair.cpp                # live synchronized image capture
│   ├── calibrate_intrinsics.cpp        # intrinsics from ChArUco images (ZED)
│   ├── dump_rs_intrinsics.cpp          # read RealSense SDK intrinsics → YAML
│   ├── calibrate_extrinsics_pair.cpp   # pairwise extrinsics via stereoCalibrate
│   ├── calibrate_zed_stereo.cpp        # ZED left↔right stereo calibration
│   ├── chain_transforms.cpp            # chain T_1_2..T_4_5 → all_to_cam1.yaml
│   ├── fuse_save_ply.cpp               # one-shot depth capture → colored PLY file
│   └── fuse_stream.cpp                 # real-time streaming point cloud viewer
├── data/
│   ├── pair_1_2/cam1/  pair_1_2/cam2/  # captured image pairs
│   ├── pair_2_3/cam2/  pair_2_3/cam3/
│   ├── pair_3_4/cam3/  pair_3_4/cam4/
│   ├── pair_4_5/cam4/  pair_4_5/cam5/
│   ├── pair_5_zed_right/cam5/  pair_5_zed_right/zed_right/
│   ├── pair_zed_right_1/zed_right/  pair_zed_right_1/cam1/
│   └── zed_stereo/zed_left/ zed_stereo/zed_right/
├── results/
│   ├── intrinsics/
│   │   ├── cam1.yaml   cam2.yaml   cam3.yaml   cam4.yaml   cam5.yaml
│   └── extrinsics/
│       ├── cam1_cam2.yaml   cam2_cam3.yaml   cam3_cam4.yaml   cam4_cam5.yaml
│       ├── cam5_zed_right.yaml   zed_right_cam1.yaml
│       ├── all_to_cam1.yaml        # optimized poses (cam1-cam5)
│       └── all_to_cam1_extra.yaml   # includes cam3_right
└── build/                              # all compiled binaries
```

---

## Calibration Board

All calibration tools use a single ChArUco board (physical board "ChArUco-400"):

| Parameter      | Value                  |
|----------------|------------------------|
| Grid size      | 9 × 12 squares         |
| Square size    | 30 mm                  |
| Marker size    | 22.5 mm                |
| Dictionary     | DICT_5X5_250           |
| Inner corners  | 88 (IDs 0 – 87)        |

Defined as constants in `include/calib_common.hpp`.

---

## Step 1 — Camera Configuration

`configs/cameras.yaml`:
```yaml
cameras:
  cam1: { model: "L515", serial: "f0210390",    role: "reference" }
  cam2: { model: "L515", serial: "f0171709" }
  cam3: { model: "ZED",  serial: "N/A", dev: 12 }
  cam4: { model: "D405", serial: "130322271201" }
  cam5: { model: "D405", serial: "218622270381" }
```

The `dev: 12` field tells `capture_pair` which `/dev/videoN` device the ZED uses.
ZED outputs a 2560×720 side-by-side frame; the tools split it at the midpoint.

---

## Step 2 — Image Capture (`capture_pair`)

Captures synchronized image pairs for each adjacent camera pair.

```bash
# From multi_cam_calib/ directory
./build/capture_pair --pair cam1 cam2
./build/capture_pair --pair cam2 cam3   # cam3 = ZED left via V4L2
./build/capture_pair --pair cam3 cam4
./build/capture_pair --pair cam4 cam5
./build/capture_pair --pair zed_left zed_right   # ZED internal stereo
```

**Key bindings during capture:**
- `SPACE` — save a pair (only when ChArUco detected in **both** cameras, ≥6 corners)
- `Q` — quit

**Live preview** shows:
- Cyan boxes: ArUco marker boundaries
- Green dots: ChArUco inner corners
- Status bar: detection count per camera

**Output:** `data/pair_X_Y/<camName>/NNNN.png` (zero-padded index).
Aim for ~40 pairs per camera combination with varied board positions (angles, distances,
all corners of the image covered).

**ZED specifics:** `capture_pair` accesses ZED as a standard V4L2 device — no ZED SDK
needed. The `--zed_dev` flag overrides the device index if needed.

---

## Step 3 — Intrinsic Calibration

### RealSense cameras (cam1, cam2, cam4, cam5)

RealSense factory intrinsics are read directly from the SDK and saved to YAML.
**Must specify capture resolution** — do not use default, which returns 1920×1080.

```bash
./build/dump_rs_intrinsics --camera cam1 --width 1280 --height 720
./build/dump_rs_intrinsics --camera cam2 --width 1280 --height 720
./build/dump_rs_intrinsics --camera cam4 --width 1280 --height 720
./build/dump_rs_intrinsics --camera cam5 --width 1280 --height 720
```

Output goes to `results/intrinsics/<name>.yaml` containing K (3×3), D (1×5),
image size, depth K/D, depth scale, and `rms: -1` (SDK-provided, not calibrated).

**Important:** If you dump at the wrong resolution and then calibrate extrinsics with
images at 1280×720, the K matrix will not match and extrinsic RMS will be ~42px instead
of sub-pixel. Always match the resolution used during `capture_pair`.

#### Calibrated intrinsics results

| Camera | fx     | fy     | cx    | cy    | Resolution | Source |
|--------|--------|--------|-------|-------|------------|--------|
| cam1   | 906.01 | 906.16 | 645.1 | 372.0 | 1280×720   | RS SDK |
| cam2   | 912.05 | 912.57 | 641.3 | 355.8 | 1280×720   | RS SDK |
| cam4   | 651.70 | 651.70 | 323.5 | 238.5 | 640×480    | OpenCV calibrateCamera (SDK K as initial guess) |
| cam5   | 645.60 | 645.60 | 323.5 | 238.5 | 640×480    | OpenCV calibrateCamera (SDK K as initial guess) |

**D405 intrinsic calibration note:** D405 cameras report `distortion.inverse_brown_conrady`
model, which is NOT compatible with OpenCV's standard Brown-Conrady model. Passing SDK
distortion coefficients directly causes 155px+ stereo RMS. The fix is to re-calibrate
intrinsics with `cv2.calibrateCamera` using `CALIB_USE_INTRINSIC_GUESS` with the SDK K
matrix as initial guess. This produces OpenCV-compatible distortion coefficients (~4px RMS).

### ZED camera (cam3)

ZED intrinsics are loaded from the **factory calibration file** (`SN17307267.conf`) rather
than re-calibrated. The factory file provides per-resolution intrinsics for left and right
lenses.

```
results/intrinsics/cam3_left.yaml   — fx=697.8, fy=697.8, cx=661.9, cy=362.8 (1280×720)
results/intrinsics/cam3_right.yaml  — fx=700.3, fy=700.3, cx=672.8, cy=358.7 (1280×720)
```

**Known issue — ZED calibration divergence:** `cv::calibrateCamera` with standard distortion
model converges to a wrong local minimum. `cv::fisheye::calibrate` crashes with
`norm_u1 > 0` assertion. Factory calibration is the practical workaround.

---

## Step 4 — Pairwise Extrinsic Calibration

Two tools are available for pairwise extrinsics:

### C++ tool: `calibrate_extrinsics_pair`

The original C++ tool. Uses `cv::stereoCalibrate` with `CALIB_FIX_INTRINSIC`.

```bash
./build/calibrate_extrinsics_pair \
    --images_a data/pair_1_2/cam1 --intr_a results/intrinsics/cam1.yaml \
    --images_b data/pair_1_2/cam2 --intr_b results/intrinsics/cam2.yaml \
    --name cam1_cam2
```

### Python tool: `calibrate_filtered.py` (recommended)

Python script with **iterative outlier removal**: runs stereoCalibrate, computes per-frame
reprojection error, removes frames above threshold, repeats up to 5 iterations. Produces
much cleaner results than single-pass calibration.

```bash
python3 calibrate_filtered.py cam1 cam2
python3 calibrate_filtered.py cam2 cam3
python3 calibrate_filtered.py cam3 cam4
python3 calibrate_filtered.py cam4 cam5
python3 calibrate_filtered.py cam5 zed_right
python3 calibrate_filtered.py zed_right cam1

# Options:
#   --threshold 15    # per-frame RMS outlier threshold (default: 15px)
#   --min_common 15   # minimum common charuco corners per frame (default: 15)
```

Camera name mapping: `cam3` → uses `cam3_left` intrinsics; `zed_right` → uses `cam3_right`.

**Transform convention:** `T_a_b` maps a point from camera B into camera A's frame:
```
p_camA = T_a_b * p_camB
```

**Critical bug fixed:** `cv::stereoCalibrate` returns R,T mapping camA→camB (i.e. T_b_from_a).
The pipeline expects T_a_b (from B into A), so the result must be **inverted** before saving.
Both the C++ and Python tools now apply `np.linalg.inv(T44)` / `cv::invert(T44)`.

### Loop closure topology

Instead of a simple chain (cam1→cam2→cam3→cam4→cam5), a **loop closure** approach is used.
Six pairwise calibrations form a loop through the ZED stereo pair:

```
cam1 ─── cam2 ─── cam3_left ─── cam4 ─── cam5 ─── cam3_right ─── cam1
                        └──── ZED factory baseline ────┘
```

Additional captured pairs:
- `pair_5_zed_right/` — cam5 ↔ ZED right
- `pair_zed_right_1/` — ZED right ↔ cam1

#### Current pairwise extrinsics results

| Pair              | Frames | RMS (px) | Translation (m)              |
|-------------------|--------|----------|------------------------------|
| cam1_cam2         | 27     | 6.58     | [-0.049, +0.000, -0.004]     |
| cam2_cam3         | 21     | 8.32     | [+0.005, -0.033, +0.013]     |
| cam3_cam4         | 15     | 14.18    | [-0.120, +0.022, +0.056]     |
| cam4_cam5         | 22     | 14.45    | [-0.029, -0.003, +0.023]     |
| cam5_zed_right    | —      | 16.50    | —                            |
| zed_right_cam1    | —      | 19.98    | —                            |

---

## Step 5 — Pose Graph Optimization (`optimize_poses.py`)

Replaces simple transform chaining with a **pose graph optimizer** that uses all pairwise
calibrations (including loop closure edges) to find globally consistent transforms.

```bash
python3 optimize_poses.py
# Output: results/extrinsics/all_to_cam1.yaml
#         results/extrinsics/all_to_cam1_extra.yaml (includes cam3_right)
```

**How it works:**
- 6 camera nodes: cam1(ref), cam2, cam3_left, cam4, cam5, cam3_right
- 7 edges: 6 pairwise calibrations + ZED factory stereo baseline (weight=5.0)
- cam1 fixed at identity; other 5 cameras have 6 DOF each (30 parameters total)
- Residuals: per-edge rotation error (rotvec) + translation error, weighted by 1/RMS
- Solver: `scipy.optimize.least_squares` with Levenberg-Marquardt
- Initial guess: chain traversal through edges

**ZED factory stereo constraint:** loaded from `SN17307267.conf` — baseline=62.9mm,
with small factory-calibrated rotation. Given weight=5.0 (high confidence).

#### Current optimized poses (T_1_X)

| Camera     | tx (m)  | ty (m)  | tz (m)  |
|------------|---------|---------|---------|
| cam1       | +0.000  | +0.000  | +0.000  |
| cam2       | -0.037  | +0.004  | -0.002  |
| cam3_left  | -0.021  | +0.042  | -0.015  |
| cam4       | +0.112  | +0.054  | +0.006  |
| cam5       | +0.078  | +0.082  | +0.013  |
| cam3_right | -0.084  | +0.041  | -0.016  |

#### Per-edge residuals after optimization

| Edge                  | Rot err | Trans err |
|-----------------------|---------|-----------|
| cam1 → cam2           | 0.50°   | 13.3 mm   |
| cam2 → cam3_left      | 0.82°   | 21.2 mm   |
| cam3_left → cam4      | 1.63°   | 24.2 mm   |
| cam4 → cam5           | 4.95°   | 19.9 mm   |
| cam5 → cam3_right     | 2.15°   | 32.7 mm   |
| cam3_right → cam1     | 4.58°   | 122.6 mm  |
| cam3_left → cam3_right| 0.00°   | 0.0 mm    |

The old `chain_transforms` C++ tool is still available but no longer the recommended path.

---

## Step 6 — ZED Stereo Calibration

ZED stereo baseline is now handled via the **factory calibration file** (`SN17307267.conf`)
loaded directly in `optimize_poses.py`. No need to run `calibrate_zed_stereo`.

The `calibrate_zed_stereo` C++ tool exists but has unresolved issues (fisheye model crash,
standard model divergence). Factory calibration is more reliable.

---

## Step 7 — Point Cloud Snapshot (`fuse_save_ply`)

Captures one synchronized frame from cam1/cam2/cam4/cam5, unprojects depth to 3D,
transforms each cloud to cam1 frame, and saves a colored ASCII PLY file.
Does not require Open3D or ZED SDK.

```bash
./build/fuse_save_ply
./build/fuse_save_ply --out results/fused.ply --max_depth 3.0
```

**How it works:**
1. Opens each RealSense camera, aligns depth to color stream
2. For each valid depth pixel: calls `rs2_deproject_pixel_to_point` using the
   **color stream intrinsics** (post-alignment)
3. Applies the 4×4 transform: `p_cam1 = T_1_N * p_camN`
4. Writes ASCII PLY with `x y z r g b` per point

Output PLY can be opened in MeshLab, CloudCompare, or any PLY viewer.

---

## Step 8 — Real-Time Streaming Viewer (`fuse_stream`)

Streams live point clouds from selected cameras, fused into cam1 frame, rendered with
Open3D's `Visualizer` class.

### Usage

```bash
# Step by step — start with one camera to verify
./build/fuse_stream --cameras cam1

# Add cam2
./build/fuse_stream --cameras cam1,cam2

# All four (if USB bandwidth allows)
./build/fuse_stream --cameras cam1,cam2,cam4,cam5

# Tune performance
./build/fuse_stream --cameras cam1,cam2 --stride 6
./build/fuse_stream --cameras cam1,cam2 --stride 8    # lightest load
./build/fuse_stream --cameras cam1,cam2 --voxel 0.01  # 1 cm voxel filter
./build/fuse_stream --max_depth 2.5                   # limit depth range
```

### Parameters

| Flag            | Default        | Description                              |
|-----------------|----------------|------------------------------------------|
| `--cameras`     | cam1,cam2,cam4,cam5 | Comma-separated list of cameras     |
| `--stride`      | 4              | Process every Nth pixel (higher = fewer points, faster) |
| `--min_depth`   | 0.25 m         | Ignore depth below this (L515 min=0.25m) |
| `--max_depth`   | 4.0 m          | Ignore depth above this                  |
| `--voxel`       | 0 (off)        | Voxel downsample size in metres          |
| `--res`         | auto           | Stream resolution WxH (e.g. `640x480`)   |
| `--fps`         | 15             | Stream framerate                         |
| `--config`      | configs/cameras.yaml | Camera serial config                |
| `--transforms`  | results/extrinsics/all_to_cam1_extra.yaml | Optimized poses |
| `--no_zed`      | off            | Skip ZED camera image plane              |
| `--no_frustums` | off            | Skip camera frustum wireframes           |
| `--t265`        | off            | Enable T265 tracking (axes + trajectory) |
| `--zed_dev`     | from config    | Override ZED V4L2 device index           |
| `--zed_scale`   | 0.5            | ZED texture resolution scale             |
| `--plane_depth` | 1.5 m          | ZED image plane distance from camera     |
| `--plane_scale` | 0.42           | ZED image plane size scale (1.0=full)    |

**USB bandwidth tip:** When running all 4 cameras, use `--res 640x480 --fps 15` to reduce
bandwidth. L515 cameras that don't support the requested resolution will fall back to auto.

### Per-camera colors

| Camera | Color   |
|--------|---------|
| cam1   | Red     |
| cam2   | Green   |
| cam4   | Yellow  |
| cam5   | Magenta |

### Features

- **Point cloud fusion:** 4 RealSense cameras (cam1/cam2/cam4/cam5) fused into cam1 frame
- **ZED image plane:** Textured quad showing ZED left camera RGB, projected at calibrated position
  - Size/distance adjustable via `--plane_depth` and `--plane_scale`
- **Camera frustums:** Cylinder-based wireframe frustums for all 6 camera positions (cam1, cam2, cam3L, cam3R, cam4, cam5), toggleable with `--no_frustums`
- **T265 tracking:** Independent visualization showing moving coordinate axes at T265 pose + trajectory trace colored by confidence (green=high, red=low). Does NOT transform other geometry (T265-to-camera extrinsics not calibrated).

### Architecture

```
main loop (Open3D PollEvents)
  ├── T265 pose update (if --t265)
  │     ├── update coordinate frame at current pose
  │     └── append trajectory trace point (colored by confidence)
  ├── appendCam() × N active RealSense cameras
  │     ├── wait_for_frames(200ms)
  │     ├── align depth → color frame
  │     └── for each pixel at stride:
  │           rs2_deproject_pixel_to_point → apply T_1_N → append
  ├── UpdateGeometry(cloud)
  ├── ZED V4L2 frame → update image plane texture
  └── ResetViewPoint(true)  [first frame only — fits view to data]
```

**Viewer controls (Open3D standard):**
- Left drag: rotate
- Right drag / scroll: zoom
- Ctrl + drag: pan
- Close window: stop

### Known issues and fixes applied

**Issue 1 — Window showed only axes, no point cloud.**
Cause: `ResetViewPoint` was never called after the first real frame; the camera was
locked to the empty bounding box. Fix: added `vis.ResetViewPoint(true)` on the first
frame with non-empty cloud.

**Issue 2 — cam5 hangs on `pipe.start()` when all 4 cameras open simultaneously.**
Cause: USB bandwidth saturation. `pipe.start()` does not throw — it blocks indefinitely.
Fix: each camera is opened in a detached `std::thread`. The caller waits with a 6-second
`condition_variable::wait_for`. If the thread does not complete in time, the camera is
skipped and the thread keeps the shared state alive via `shared_ptr` until it eventually
returns (or the process exits).

**Issue 3 — `failed to set power state` error.**
Cause: creating an `rs2::context` for device enumeration AND then starting pipelines
caused context conflicts. Fix: enumerate devices once with a single context before any
pipelines are created, then destroy that context before calling `pipe.start()`.

**Issue 4 — Wayland warning `cannot set window position`.**
This is harmless. Open3D's GLFW backend cannot set the initial window position on Wayland.
The window still opens and renders correctly via XWayland.

---

## Build Instructions

```bash
cd multi_cam_calib/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

All binaries land in `build/`. Run commands from the `multi_cam_calib/` directory so
relative paths (`configs/`, `data/`, `results/`) resolve correctly.

**Targets built:**

| Binary                     | Dependencies              |
|----------------------------|---------------------------|
| `calibrate_intrinsics`     | OpenCV                    |
| `calibrate_extrinsics_pair`| OpenCV                    |
| `chain_transforms`         | OpenCV                    |
| `calibrate_zed_stereo`     | OpenCV                    |
| `dump_rs_intrinsics`       | librealsense2 + OpenCV    |
| `capture_pair`             | librealsense2 + OpenCV    |
| `fuse_save_ply`            | librealsense2 + OpenCV    |
| `fuse_stream`              | librealsense2 + OpenCV + Open3D |
| `fuse_and_view`            | SKIPPED (needs ZED SDK + Open3D) |

---

## Issues Resolved

1. **Transform inversion bug (CRITICAL):** `cv::stereoCalibrate` returns R,T mapping
   camA→camB (T_b_from_a), but the pipeline expects T_a_b (from B into A). Both the C++
   and Python calibration tools now invert the result before saving. This was the root cause
   of the initial "big gap" in the fused point cloud.

2. **D405 inverse_brown_conrady distortion:** D405 cameras use a distortion model
   incompatible with OpenCV. Passing SDK coefficients directly caused 155px+ stereo RMS.
   Fixed by re-calibrating D405 intrinsics with `cv2.calibrateCamera` using
   `CALIB_USE_INTRINSIC_GUESS` with SDK K as initial guess (→ ~4px RMS, correct model).

3. **ZED placeholder intrinsics:** cam3.yaml had estimated fx=674 with zero distortion.
   Replaced with factory calibration from `SN17307267.conf` (fx=697.8 for left, fx=700.3
   for right at 1280×720).

4. **cam4_cam5 unreliable:** Original calibration had RMS=101px and T~2m (unrealistic).
   Recaptured and recalibrated to RMS=14.5px with sensible translation.

5. **Upper-lower camera misalignment:** The chain cam2→cam3→cam4 was the weak link between
   upper cameras (cam1/cam2) and lower cameras (cam4/cam5). Improved by:
   - Loop closure topology with pose graph optimization
   - ZED factory stereo baseline as high-confidence constraint
   - Multiple rounds of recapture for cam2-cam3 and cam3-cam4 pairs
   - D405 intrinsic recalibration

6. **USB bandwidth:** Added `--res` and `--fps` flags to `fuse_stream` with automatic
   fallback for cameras that don't support the requested resolution.

## Pending Work

1. **cam3_right→cam1 loop closure residual** — 4.6° rotation / 123mm translation error.
   Could be improved with recapture of this pair.

2. **cam3 (ZED) depth in fuse_stream** — currently only cam1/cam2/cam4/cam5 are fused.
   Would require ZED SDK or V4L2 depth access to include cam3.

3. **T265-to-camera extrinsic calibration** — T265 tracking is shown independently (axes +
   trace) because the rigid transform between T265 and the camera array is unknown. Options:
   - Physical measurement from 3D-printed rig CAD
   - Hand-eye calibration (limited by T265 V-SLAM drift ~150mm)
   - Currently NOT applied to scene geometry

4. **Install ZED SDK** (optional) — would enable cam3 depth in `fuse_stream` and allow
   `fuse_and_view` to build.