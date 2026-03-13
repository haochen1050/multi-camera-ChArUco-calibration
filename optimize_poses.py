#!/usr/bin/env python3
"""
Pose graph optimization for multi-camera extrinsic calibration.
Uses all pairwise stereo calibrations (including loop edges) to find
globally consistent transforms that minimize weighted reprojection error.

Cameras: cam1 (reference), cam2, cam3_left, cam4, cam5, cam3_right
Edges: cam1-cam2, cam2-cam3(left), cam3(left)-cam4, cam4-cam5,
       cam5-cam3(right), cam3(right)-cam1

Additionally uses the known ZED stereo baseline (cam3_left <-> cam3_right)
as a hard constraint.
"""
import cv2, numpy as np, os, sys
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# ── Load pairwise transforms ─────────────────────────────────────────────────

def load_T44(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    T = fs.getNode("T_a_b").mat()
    if T is None:
        raise RuntimeError(f"Cannot load T_a_b from {path}")
    rms_node = fs.getNode("rms")
    rms = rms_node.real() if rms_node is not None else 999.0
    return T.astype(np.float64), rms

# Camera nodes: cam1=0, cam2=1, cam3_left=2, cam4=3, cam5=4, cam3_right=5
CAM_NAMES = ["cam1", "cam2", "cam3_left", "cam4", "cam5", "cam3_right"]
N_CAMS = len(CAM_NAMES)

# Edge definitions: (camA_idx, camB_idx, yaml_file)
# T_a_b in the YAML maps FROM camB INTO camA
EDGES = [
    (0, 1, "results/extrinsics/cam1_cam2.yaml"),        # cam1-cam2
    (1, 2, "results/extrinsics/cam2_cam3.yaml"),         # cam2-cam3_left
    (2, 3, "results/extrinsics/cam3_cam4.yaml"),         # cam3_left-cam4
    (3, 4, "results/extrinsics/cam4_cam5.yaml"),         # cam4-cam5
    (4, 5, "results/extrinsics/cam5_zed_right.yaml"),    # cam5-cam3_right
    (5, 0, "results/extrinsics/zed_right_cam1.yaml"),    # cam3_right-cam1
]

# Load all edges
edges = []
for a, b, path in EDGES:
    if not os.path.exists(path):
        print(f"[WARN] Missing {path}, skipping edge {CAM_NAMES[a]}-{CAM_NAMES[b]}")
        continue
    T, rms = load_T44(path)
    # Weight: inverse of RMS (better calibrations have more influence)
    weight = 1.0 / max(rms, 1.0)
    edges.append((a, b, T, weight))
    print(f"  Edge {CAM_NAMES[a]:>10s} -> {CAM_NAMES[b]:>10s}  RMS={rms:6.2f}px  weight={weight:.3f}")

# ZED stereo constraint: cam3_left <-> cam3_right from factory calibration
# Baseline=62.9444mm, small rotation
baseline = 0.0629444  # meters
# From factory: RX=-0.000469, RZ=0.000540, CV(=RY)=0.003628
rx, ry, rz = -0.000469092, 0.00362766, 0.000539973
R_lr = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
T_zed_left_right = np.eye(4)
T_zed_left_right[:3, :3] = R_lr
T_zed_left_right[:3, 3] = [baseline, -0.00015746, -0.000394451]  # TY, TZ from factory
# This is T mapping from right to left: p_left = T * p_right
# But we need to check convention. ZED baseline is left-to-right distance.
# The factory T is the transform from left camera to right camera.
# So we invert to get "from right into left" matching our convention.
T_zed_lr = np.linalg.inv(T_zed_left_right)
zed_weight = 5.0  # high confidence in factory calibration
edges.append((2, 5, T_zed_lr, zed_weight))
print(f"  Edge {'cam3_left':>10s} -> {'cam3_right':>10s}  (ZED factory)  weight={zed_weight:.1f}")

print(f"\nTotal edges: {len(edges)}")

# ── Parameterization ─────────────────────────────────────────────────────────
# cam1 is fixed at identity. Other 5 cameras have 6 DOF each (rotvec + translation).
# Total parameters: 5 * 6 = 30

def params_to_poses(x):
    """Convert parameter vector to list of 4x4 transforms (cam1=identity)."""
    poses = [np.eye(4)]  # cam1 = identity
    for i in range(N_CAMS - 1):
        p = x[i*6 : (i+1)*6]
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p[3:6]
        poses.append(T)
    return poses

def residuals(x):
    """Compute weighted residuals for all edges."""
    poses = params_to_poses(x)
    res = []
    for a, b, T_ab_measured, w in edges:
        # T_ab_measured maps FROM b INTO a: p_a = T_ab * p_b
        # From our poses: T_1_a and T_1_b (both map from camX into cam1)
        # Predicted T_ab = inv(T_1_a) * T_1_b ... no wait
        # T_1_a: from a into cam1. T_1_b: from b into cam1.
        # T_ab: from b into a = inv(T_1_a) * T_1_b
        T_1_a = poses[a]
        T_1_b = poses[b]
        T_ab_predicted = np.linalg.inv(T_1_a) @ T_1_b

        # Error: difference between predicted and measured
        # Use rotation error (angle) + translation error
        dT = np.linalg.inv(T_ab_measured) @ T_ab_predicted

        # Rotation error as rotation vector (3 components)
        dR = Rotation.from_matrix(dT[:3, :3]).as_rotvec()

        # Translation error (3 components)
        dt = dT[:3, 3]

        # Weighted residuals (6 per edge)
        res.extend((w * dR).tolist())
        res.extend((w * dt).tolist())

    return np.array(res)

# ── Initial guess: chain the first 5 edges ───────────────────────────────────
print("\n=== Building initial guess from chain ===")
# Start from cam1=identity, chain through edges
init_poses = [np.eye(4)] * N_CAMS

# Chain: cam1 -> cam2 -> cam3_left -> cam4 -> cam5
chain_order = [(0, 1), (1, 2), (2, 3), (3, 4)]
for a, b in chain_order:
    # Find the edge
    for ea, eb, T_meas, w in edges:
        if ea == a and eb == b:
            # T_meas maps from b into a. T_1_b = T_1_a * T_a_b = T_1_a * T_meas
            # Wait: T_a_b maps from b into a means p_a = T_a_b * p_b
            # We want T_1_b = T_1_a * T_a_b^(-1)? No.
            # T_1_a maps from a into cam1: p_1 = T_1_a * p_a
            # T_a_b maps from b into a: p_a = T_a_b * p_b
            # So p_1 = T_1_a * T_a_b * p_b => T_1_b = T_1_a * T_a_b
            init_poses[b] = init_poses[a] @ T_meas
            break

# cam3_right: use cam5 -> cam3_right edge
for ea, eb, T_meas, w in edges:
    if ea == 4 and eb == 5:  # cam5 -> cam3_right
        init_poses[5] = init_poses[4] @ T_meas
        break

# Build initial parameter vector
x0 = []
for i in range(1, N_CAMS):
    rv = Rotation.from_matrix(init_poses[i][:3, :3]).as_rotvec()
    t = init_poses[i][:3, 3]
    x0.extend(rv.tolist())
    x0.extend(t.tolist())
x0 = np.array(x0)

print("Initial poses (before optimization):")
for i, name in enumerate(CAM_NAMES):
    t = init_poses[i][:3, 3]
    print(f"  {name:>10s}  t=[{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]")

init_res = residuals(x0)
print(f"Initial cost: {np.sum(init_res**2):.6f}")

# ── Optimize ──────────────────────────────────────────────────────────────────
print("\n=== Optimizing pose graph ===")
result = least_squares(residuals, x0, method='lm', max_nfev=10000,
                       ftol=1e-12, xtol=1e-12, gtol=1e-12)
print(f"  Status: {result.message}")
print(f"  Final cost: {result.cost:.6f}  (initial: {np.sum(init_res**2)/2:.6f})")

opt_poses = params_to_poses(result.x)

print("\nOptimized poses (T_1_X):")
for i, name in enumerate(CAM_NAMES):
    t = opt_poses[i][:3, 3]
    print(f"  {name:>10s}  t=[{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]")

# ── Check loop closure error ─────────────────────────────────────────────────
print("\n=== Per-edge residuals ===")
for a, b, T_meas, w in edges:
    T_1_a = opt_poses[a]
    T_1_b = opt_poses[b]
    T_ab_pred = np.linalg.inv(T_1_a) @ T_1_b
    dT = np.linalg.inv(T_meas) @ T_ab_pred
    angle = np.degrees(np.linalg.norm(Rotation.from_matrix(dT[:3,:3]).as_rotvec()))
    dt = np.linalg.norm(dT[:3, 3]) * 1000  # mm
    print(f"  {CAM_NAMES[a]:>10s}-{CAM_NAMES[b]:<10s}  rot_err={angle:.2f}°  trans_err={dt:.1f}mm")

# ── Save results ──────────────────────────────────────────────────────────────
# Map cam3_left -> cam3 for compatibility with fuse_stream
cam_map = {"cam1": "1", "cam2": "2", "cam3_left": "3", "cam4": "4", "cam5": "5"}

out_path = "results/extrinsics/all_to_cam1.yaml"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
for i, name in enumerate(CAM_NAMES):
    if name not in cam_map:
        continue
    key = f"T_1_{cam_map[name]}"
    fs.write(key, opt_poses[i])
fs.release()
print(f"\n[SAVE] {out_path}")

# Also save cam3_right pose for reference
out_extra = "results/extrinsics/all_to_cam1_extra.yaml"
fs = cv2.FileStorage(out_extra, cv2.FILE_STORAGE_WRITE)
for i, name in enumerate(CAM_NAMES):
    fs.write(f"T_1_{name}", opt_poses[i])
fs.release()
print(f"[SAVE] {out_extra} (includes cam3_right)")
