# Multi-Camera ChArUco Calibration

Extrinsic calibration and real-time point cloud fusion for a 5-camera RGBD array with 6DoF pose tracking.

## Camera Array

| Camera | Model      | Serial         | Notes                    |
|--------|------------|----------------|--------------------------|
| cam1   | Intel L515 | f0210390       | Reference frame          |
| cam2   | Intel L515 | f0171709       |                          |
| cam3   | ZED-M      | SN17307267     | Stereo pair (V4L2, no SDK) |
| cam4   | Intel D405 | 130322271201   |                          |
| cam5   | Intel D405 | 218622270381   |                          |
| t265   | Intel T265 | 905312111793   | 6DoF pose tracking       |

## Dependencies

- **OpenCV** 4.6+ (ArUco module)
- **librealsense2** (RealSense cameras + T265)
- **Open3D** 0.19+ (real-time viewer)
- **Python 3** with NumPy, SciPy, OpenCV (`calibrate_filtered.py`, `optimize_poses.py`)
- ZED SDK is **not required** — ZED is accessed via V4L2

## Build

```bash
cd multi_cam_calib/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Calibration Pipeline

### 1. Dump RealSense Intrinsics

```bash
./build/dump_rs_intrinsics --camera cam1 --width 1280 --height 720
./build/dump_rs_intrinsics --camera cam2 --width 1280 --height 720
./build/dump_rs_intrinsics --camera cam4 --width 1280 --height 720
./build/dump_rs_intrinsics --camera cam5 --width 1280 --height 720
```

ZED intrinsics are loaded from the factory calibration file (`SN17307267.conf`).

### 2. Capture Image Pairs

```bash
./build/capture_pair --pair cam1 cam2
./build/capture_pair --pair cam2 cam3
./build/capture_pair --pair cam3 cam4
./build/capture_pair --pair cam4 cam5
./build/capture_pair --pair cam5 zed_right
./build/capture_pair --pair zed_right cam1
```

Press **SPACE** to save a pair (requires ChArUco detection in both views), **Q** to quit.

### 3. Pairwise Extrinsic Calibration

```bash
python3 calibrate_filtered.py cam1 cam2
python3 calibrate_filtered.py cam2 cam3
python3 calibrate_filtered.py cam3 cam4
python3 calibrate_filtered.py cam4 cam5
python3 calibrate_filtered.py cam5 zed_right
python3 calibrate_filtered.py zed_right cam1
```

Uses iterative outlier removal for robust stereo calibration.

### 4. Pose Graph Optimization

```bash
python3 optimize_poses.py
```

Optimizes all 6 pairwise edges + ZED factory stereo baseline into globally consistent transforms. Output: `results/extrinsics/all_to_cam1.yaml`.

### Camera Graph

```
cam1 --- cam2 --- cam3_left --- cam4 --- cam5 --- cam3_right --- cam1
                       |___ ZED stereo baseline ___|
```

## Real-Time Viewer

```bash
# Basic usage
./build/fuse_stream --cameras cam1,cam2,cam4,cam5 --res 640x480

# With T265 pose tracking visualization
./build/fuse_stream --cameras cam1,cam2,cam4,cam5 --res 640x480 --t265

# With T265-driven scene rotation
./build/fuse_stream --cameras cam1,cam2,cam4,cam5 --res 640x480 --rotate_frustums
```

### Flags

| Flag              | Default                | Description                                    |
|-------------------|------------------------|------------------------------------------------|
| `--cameras`       | cam1,cam2,cam4,cam5    | Comma-separated camera list                    |
| `--stride`        | 4                      | Process every Nth pixel                        |
| `--min_depth`     | 0.25 m                 | Min depth for L515 (D405 fixed at 0.07m)       |
| `--max_depth`     | 4.0 m                  | Max depth for L515 (D405 capped at 0.5m)       |
| `--voxel`         | 0 (off)                | Voxel downsample size (metres)                 |
| `--res`           | auto                   | Stream resolution, e.g. `640x480`              |
| `--fps`           | 30                     | Stream framerate                               |
| `--frustums`      | off                    | Show camera frustum wireframes                 |
| `--t265`          | off                    | Show T265 axes + trajectory trace              |
| `--rotate_frustums` | off                  | Rotate scene with T265 6DoF data               |
| `--no_zed`        | off                    | Skip ZED image plane                           |
| `--plane_depth`   | 1.5 m                  | ZED image plane distance                       |
| `--plane_scale`   | 0.42                   | ZED image plane size                           |

### Controls

- **Left drag** — rotate view
- **Right drag / scroll** — zoom
- **Ctrl + drag** — pan
- **Q / close window** — stop

## ChArUco Board

| Parameter    | Value          |
|--------------|----------------|
| Grid         | 9 x 12 squares |
| Square size  | 30 mm          |
| Marker size  | 22.5 mm        |
| Dictionary   | DICT_5X5_250   |

## Project Structure

```
multi_cam_calib/
├── CMakeLists.txt
├── calibrate_filtered.py          # Stereo calibration with outlier removal
├── optimize_poses.py              # Pose graph optimization
├── configs/cameras.yaml           # Camera serials and device config
├── include/calib_common.hpp       # Shared helpers (board, ArUco, transforms)
├── src/
│   ├── capture_pair.cpp           # Synchronized image capture
│   ├── dump_rs_intrinsics.cpp     # RealSense SDK intrinsics to YAML
│   ├── calibrate_intrinsics.cpp   # Intrinsics from ChArUco images
│   ├── calibrate_extrinsics_pair.cpp  # Pairwise stereo calibration
│   ├── fuse_save_ply.cpp          # One-shot depth capture to PLY
│   └── fuse_stream.cpp            # Real-time streaming viewer
├── data/                          # Captured image pairs
└── results/
    ├── intrinsics/                # Per-camera K, D, resolution
    └── extrinsics/                # Pairwise and optimized transforms
```
