#!/usr/bin/env python3
"""
Stereo calibration with outlier filtering.
Runs stereoCalibrate, identifies per-frame outliers, removes them, and reruns.
Saves result in the same YAML format as calibrate_extrinsics_pair.
"""
import cv2, numpy as np, glob, os, sys

SQUARES_X, SQUARES_Y = 9, 12
SQUARE_LEN, MARKER_LEN = 0.030, 0.0225
DICT_ID = cv2.aruco.DICT_5X5_250

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_ID)
board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LEN, MARKER_LEN, aruco_dict)
params = cv2.aruco.DetectorParameters_create()

def detect(img_path, K, D):
    img = cv2.imread(img_path)
    if img is None: return [], []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    if ids is None or len(ids) == 0: return [], []
    _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board, cameraMatrix=K, distCoeffs=D)
    if ch_ids is None or len(ch_ids) < 15: return [], []
    return ch_corners.reshape(-1, 2), ch_ids.flatten()

def load_intr(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat().astype(np.float64)
    D = fs.getNode("D").mat().astype(np.float64)
    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    return K, D, (w, h)

def per_frame_rms_b(obj_pts, img_a, img_b, KA, DA, KB, DB, R, T):
    """Compute per-frame stereo projection RMS (project cam_a pose into cam_b)."""
    rms_list = []
    for i in range(len(obj_pts)):
        ok, rvec, tvec = cv2.solvePnP(obj_pts[i], img_a[i], KA, DA)
        if not ok:
            rms_list.append(999)
            continue
        R_mat, _ = cv2.Rodrigues(rvec)
        R2 = R @ R_mat
        T2 = R @ tvec + T
        proj, _ = cv2.projectPoints(obj_pts[i], cv2.Rodrigues(R2)[0], T2, KB, DB)
        rms = float(np.sqrt(np.mean((img_b[i] - proj.reshape(-1, 2))**2)))
        rms_list.append(rms)
    return rms_list

# ── CLI ──────────────────────────────────────────────────────────────────────
if len(sys.argv) < 3:
    print("Usage: calibrate_filtered.py <camA> <camB> [--threshold 15] [--min_common 15]")
    sys.exit(1)

camA, camB = sys.argv[1], sys.argv[2]
threshold = 15.0
min_common = 15
for i, a in enumerate(sys.argv):
    if a == "--threshold" and i+1 < len(sys.argv):
        threshold = float(sys.argv[i+1])
    if a == "--min_common" and i+1 < len(sys.argv):
        min_common = int(sys.argv[i+1])

def cam_short(name):
    return name.replace("cam", "") if name.startswith("cam") else name

def cam_intr(name):
    # cam3 -> cam3_left (ZED left = cam3 in old convention)
    if name == "cam3": return "cam3_left"
    return name.replace("zed_", "cam3_")  # zed_right -> cam3_right

pair_dir = f"data/pair_{cam_short(camA)}_{cam_short(camB)}"
intr_a = f"results/intrinsics/{cam_intr(camA)}.yaml"
intr_b = f"results/intrinsics/{cam_intr(camB)}.yaml"
out_path = f"results/extrinsics/{camA}_{camB}.yaml"

KA, DA, szA = load_intr(intr_a)
KB, DB, szB = load_intr(intr_b)
print(f"Intrinsics: {camA} {szA}, {camB} {szB}")

# Load images
dir_a = f"{pair_dir}/{camA}"
dir_b = f"{pair_dir}/{camB}"
files_a = sorted(glob.glob(f"{dir_a}/*.png"))
files_b = sorted(glob.glob(f"{dir_b}/*.png"))
print(f"Images: {camA}={len(files_a)}  {camB}={len(files_b)}")

# Detect and match
obj_pts, img_a, img_b, fnames = [], [], [], []
for fa, fb in zip(files_a, files_b):
    cA, iA = detect(fa, KA, DA)
    cB, iB = detect(fb, KB, DB)
    if len(cA) == 0 or len(cB) == 0: continue
    common = np.intersect1d(iA, iB)
    if len(common) < min_common: continue
    idxA = [np.where(iA == c)[0][0] for c in common]
    idxB = [np.where(iB == c)[0][0] for c in common]
    obj3d = np.array([board.chessboardCorners[c] for c in common], dtype=np.float32)
    obj_pts.append(obj3d)
    img_a.append(cA[idxA].astype(np.float32))
    img_b.append(cB[idxB].astype(np.float32))
    fnames.append(os.path.basename(fa))

print(f"Valid pairs: {len(fnames)}")

# ── Iterative outlier removal ────────────────────────────────────────────────
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

for iteration in range(5):
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_a, img_b,
        KA.copy(), DA.copy(), KB.copy(), DB.copy(),
        szA, flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)

    rms_list = per_frame_rms_b(obj_pts, img_a, img_b, KA, DA, KB, DB, R, T)

    # Find outliers
    good = [i for i, r in enumerate(rms_list) if r < threshold]
    bad = [i for i, r in enumerate(rms_list) if r >= threshold]

    if len(bad) == 0:
        break

    print(f"  Iter {iteration}: RMS={rms:.2f}px, removing {len(bad)} outliers (RMS_B > {threshold}px)")
    obj_pts = [obj_pts[i] for i in good]
    img_a = [img_a[i] for i in good]
    img_b = [img_b[i] for i in good]
    fnames = [fnames[i] for i in good]

# Final calibration
rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    obj_pts, img_a, img_b,
    KA.copy(), DA.copy(), KB.copy(), DB.copy(),
    szA, flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)

print(f"\n=== Final: {len(fnames)} frames, stereoCalibrate RMS = {rms:.3f} px ===")
print(f"T (m): [{T[0,0]:.6f}, {T[1,0]:.6f}, {T[2,0]:.6f}]")
print(f"|T| = {np.linalg.norm(T):.4f} m")

# Build 4x4: stereoCalibrate gives T_b_from_a (camA→camB).
# chain_transforms expects T_a_b = "from b into a", so we need the inverse.
T44_ab = np.eye(4, dtype=np.float64)
T44_ab[:3, :3] = R
T44_ab[:3, 3] = T.flatten()
T44 = np.linalg.inv(T44_ab)

# Save
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
fs.write("T_a_b", T44)
fs.write("rms", rms)
fs.release()
print(f"[SAVE] {out_path}")
