#!/usr/bin/env python3
"""
Relay calibration: compute relative transform between two cameras that
see the same board but with minimal/no corner overlap.

For each frame pair, solvePnP independently in each camera, then:
  T_a_b = inv(T_cam_a_board) * T_cam_b_board

This works even with zero common corners — each camera just needs to see
enough of the board independently.

Multiple frames are used to compute a robust median transform.
"""
import cv2, numpy as np, glob, os, sys
from scipy.spatial.transform import Rotation

SQUARES_X, SQUARES_Y = 9, 12
SQUARE_LEN, MARKER_LEN = 0.030, 0.0225
DICT_ID = cv2.aruco.DICT_5X5_250

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_ID)
board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LEN, MARKER_LEN, aruco_dict)
params = cv2.aruco.DetectorParameters_create()

def detect_and_solve(img_path, K, D):
    """Detect charuco and solvePnP. Returns T_cam_board (4x4) or None."""
    img = cv2.imread(img_path)
    if img is None: return None, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None or len(ids) < 4: return None, 0
    _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board, cameraMatrix=K, distCoeffs=D)
    if ch_ids is None or len(ch_ids) < 6: return None, 0

    obj_pts = np.array([board.chessboardCorners[c] for c in ch_ids.flatten()], dtype=np.float32)
    img_pts = ch_corners.reshape(-1, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D)
    if not ok: return None, 0

    # Compute reprojection error
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean((img_pts - proj.reshape(-1, 2))**2)))

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T, rms

def load_intr(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat().astype(np.float64)
    D = fs.getNode("D").mat().astype(np.float64)
    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    return K, D, (w, h)

def cam_short(name):
    return name.replace("cam", "") if name.startswith("cam") else name

def cam_intr(name):
    if name == "cam3": return "cam3_left"
    return name.replace("zed_", "cam3_")

# ── CLI ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("Usage: calibrate_relay.py <camA> <camB> [--max_rms 5]")
    sys.exit(1)

camA, camB = sys.argv[1], sys.argv[2]
max_rms = 5.0
for i, a in enumerate(sys.argv):
    if a == "--max_rms" and i+1 < len(sys.argv):
        max_rms = float(sys.argv[i+1])

pair_dir = f"data/pair_{cam_short(camA)}_{cam_short(camB)}"
intr_a = f"results/intrinsics/{cam_intr(camA)}.yaml"
intr_b = f"results/intrinsics/{cam_intr(camB)}.yaml"
out_path = f"results/extrinsics/{camA}_{camB}.yaml"

KA, DA, szA = load_intr(intr_a)
KB, DB, szB = load_intr(intr_b)
print(f"Intrinsics: {camA} {szA}, {camB} {szB}")

dir_a = f"{pair_dir}/{camA}"
dir_b = f"{pair_dir}/{camB}"
files_a = sorted(glob.glob(f"{dir_a}/*.png"))
files_b = sorted(glob.glob(f"{dir_b}/*.png"))
print(f"Images: {camA}={len(files_a)}  {camB}={len(files_b)}")

# ── Compute per-frame relative transforms ────────────────────────────────────
transforms = []
for fa, fb in zip(files_a, files_b):
    T_a, rms_a = detect_and_solve(fa, KA, DA)
    T_b, rms_b = detect_and_solve(fb, KB, DB)
    if T_a is None or T_b is None: continue
    if rms_a > max_rms or rms_b > max_rms:
        continue

    # T_a = T_camA_board, T_b = T_camB_board
    # We want T_a_b = maps from camB into camA = T_camA_board * inv(T_camB_board)
    T_ab = T_a @ np.linalg.inv(T_b)
    transforms.append(T_ab)
    fname = os.path.basename(fa)
    t = T_ab[:3, 3]
    print(f"  {fname}  rmsA={rms_a:.1f}  rmsB={rms_b:.1f}  t=[{t[0]:.4f},{t[1]:.4f},{t[2]:.4f}]")

print(f"\nValid frames: {len(transforms)}")
if len(transforms) < 3:
    print("Too few valid frames. Try --max_rms higher.")
    sys.exit(1)

# ── Compute robust average transform ─────────────────────────────────────────
# Use median translation and mean rotation (via quaternion averaging)
translations = np.array([T[:3, 3] for T in transforms])
t_median = np.median(translations, axis=0)

# Filter outliers: remove transforms where translation is far from median
dists = np.linalg.norm(translations - t_median, axis=1)
threshold = np.median(dists) * 3 + 0.01  # 3x median absolute deviation
good = dists < threshold
print(f"Inliers: {np.sum(good)} / {len(transforms)} (threshold={threshold*1000:.1f}mm)")

good_transforms = [T for T, g in zip(transforms, good) if g]
translations = np.array([T[:3, 3] for T in good_transforms])
t_final = np.median(translations, axis=0)

# Average rotation via quaternions
quats = np.array([Rotation.from_matrix(T[:3,:3]).as_quat() for T in good_transforms])
# Ensure consistent quaternion sign
for i in range(1, len(quats)):
    if np.dot(quats[i], quats[0]) < 0:
        quats[i] = -quats[i]
q_mean = np.mean(quats, axis=0)
q_mean /= np.linalg.norm(q_mean)
R_final = Rotation.from_quat(q_mean).as_matrix()

T44 = np.eye(4, dtype=np.float64)
T44[:3, :3] = R_final
T44[:3, 3] = t_final

# Report spread
t_std = np.std(translations, axis=0) * 1000
print(f"Translation spread (std): [{t_std[0]:.1f}, {t_std[1]:.1f}, {t_std[2]:.1f}] mm")
print(f"T (m): [{t_final[0]:.6f}, {t_final[1]:.6f}, {t_final[2]:.6f}]")
print(f"|T| = {np.linalg.norm(t_final):.4f} m")

# Save
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
fs.write("T_a_b", T44)
fs.write("rms", float(np.mean(t_std)))  # use spread as quality metric
fs.release()
print(f"[SAVE] {out_path}")
