// =============================================================================
//  fuse_stream.cpp
//  Real-time multi-camera point cloud streaming into Camera 1's frame.
//  Opens selected RealSense cameras, projects every depth frame into
//  cam1 coordinates and renders the merged cloud live with Open3D.
//  Also displays ZED left/right RGB as textured planes, and camera frustums.
//
//  No ZED SDK required.
//
//  Usage:
//    fuse_stream [--config     <cameras.yaml>]
//               [--transforms  <results/extrinsics/all_to_cam1_extra.yaml>]
//               [--cameras     cam1,cam2,cam4,cam5]   default: all
//               [--min_depth   <m>]  default 0.1
//               [--max_depth   <m>]  default 4.0
//               [--stride      <px>] default 4  (skip pixels; higher = faster)
//               [--voxel       <m>]  default 0  (0 = no voxel downsampling)
//               [--no_zed]           skip ZED camera
//               [--no_frustums]      skip camera frustum display
//               [--zed_dev <N>]      ZED V4L2 device index (default from config)
//               [--zed_scale <f>]    ZED texture scale (default 0.5)
//               [--plane_depth <m>]  ZED image plane depth (default 0.5)
// =============================================================================

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <array>
#include <cstring>
#include <filesystem>
#include <set>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>

#include "calib_common.hpp"

namespace o3d = open3d;
namespace fs  = std::filesystem;

// ── Per-camera state ──────────────────────────────────────────────────────────

struct RsCam {
    std::string    name;
    rs2::pipeline  pipe;
    rs2::align     align_to_color{RS2_STREAM_COLOR};
    rs2_intrinsics color_intr{};
    float          depth_scale = 0.f;
    double         T[4][4]{};             // transform to cam1 frame
    Eigen::Vector3d color;                // per-camera display color
};

static int streamW = 0, streamH = 0, streamFps = 15;

static std::unique_ptr<RsCam> openCam(const std::string& name,
                                       const std::string& serial,
                                       const Eigen::Vector3d& displayColor)
{
    auto c = std::make_unique<RsCam>();
    c->name  = name;
    c->color = displayColor;

    try {
        rs2::config cfg;
        cfg.enable_device(serial);
        // Try requested resolution; fall back to auto if unsupported
        bool started = false;
        if (streamW > 0 && streamH > 0) {
            try {
                cfg.enable_stream(RS2_STREAM_DEPTH, streamW, streamH, RS2_FORMAT_Z16,  streamFps);
                cfg.enable_stream(RS2_STREAM_COLOR, streamW, streamH, RS2_FORMAT_RGB8, streamFps);
                auto profile = c->pipe.start(cfg);
                c->depth_scale = profile.get_device()
                                        .first<rs2::depth_sensor>()
                                        .get_depth_scale();
                c->color_intr  = profile.get_stream(RS2_STREAM_COLOR)
                                        .as<rs2::video_stream_profile>().get_intrinsics();
                started = true;
            } catch (...) {
                // Resolution not supported, fall back to auto
                c->pipe = rs2::pipeline();
                cfg = rs2::config();
                cfg.enable_device(serial);
            }
        }
        if (!started) {
            cfg.enable_stream(RS2_STREAM_DEPTH, 0, 0, RS2_FORMAT_Z16,  30);
            cfg.enable_stream(RS2_STREAM_COLOR, 0, 0, RS2_FORMAT_RGB8, 30);
            auto profile = c->pipe.start(cfg);
            c->depth_scale = profile.get_device()
                                    .first<rs2::depth_sensor>()
                                    .get_depth_scale();
            c->color_intr  = profile.get_stream(RS2_STREAM_COLOR)
                                    .as<rs2::video_stream_profile>().get_intrinsics();
        }

        // Warm-up: discard early frames
        for (int i = 0; i < 10; ++i) c->pipe.wait_for_frames();

        std::cout << " ok  color=" << c->color_intr.width << "x" << c->color_intr.height
                  << "  scale=" << c->depth_scale << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[SKIP] " << name << " (" << serial << "): " << e.what() << "\n";
        return nullptr;
    }
    return c;
}

// ── Load 4×4 OpenCV matrix from YAML ─────────────────────────────────────────

static bool loadT44(const std::string& path, const std::string& key,
                    double T[4][4])
{
    cv::FileStorage fss(path, cv::FileStorage::READ);
    if (!fss.isOpened()) {
        std::cerr << "[ERROR] Cannot open: " << path << "\n";
        return false;
    }
    cv::Mat m;
    fss[key] >> m;
    if (m.empty()) {
        std::cerr << "[ERROR] Key '" << key << "' not found in " << path << "\n";
        return false;
    }
    m.convertTo(m, CV_64F);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            T[r][c] = m.at<double>(r, c);
    return true;
}

// ── Transform a 3D point by a 4×4 matrix ─────────────────────────────────────

static Eigen::Vector3d transformPt(const double T[4][4],
                                   double x, double y, double z)
{
    return {T[0][0]*x + T[0][1]*y + T[0][2]*z + T[0][3],
            T[1][0]*x + T[1][1]*y + T[1][2]*z + T[1][3],
            T[2][0]*x + T[2][1]*y + T[2][2]*z + T[2][3]};
}

// ── Unproject image corner to 3D at given depth ──────────────────────────────

static Eigen::Vector3d unprojectCorner(double u, double v,
                                       double fx, double fy, double cx, double cy,
                                       double depth,
                                       const double T[4][4])
{
    double x = (u - cx) * depth / fx;
    double y = (v - cy) * depth / fy;
    return transformPt(T, x, y, depth);
}

// ── Create a textured quad mesh for a single camera image plane ──────────────

static std::shared_ptr<o3d::geometry::TriangleMesh>
createImagePlane(const double T[4][4],
                 double fx, double fy, double cx, double cy,
                 int imgW, int imgH, double planeDepth,
                 double scale = 1.0)
{
    auto mesh = std::make_shared<o3d::geometry::TriangleMesh>();

    auto tl = unprojectCorner(0,    0,    fx, fy, cx, cy, planeDepth, T);
    auto tr = unprojectCorner(imgW, 0,    fx, fy, cx, cy, planeDepth, T);
    auto br = unprojectCorner(imgW, imgH, fx, fy, cx, cy, planeDepth, T);
    auto bl = unprojectCorner(0,    imgH, fx, fy, cx, cy, planeDepth, T);

    // Shrink corners toward center while keeping same distance
    if (scale != 1.0) {
        Eigen::Vector3d center = (tl + tr + br + bl) / 4.0;
        tl = center + scale * (tl - center);
        tr = center + scale * (tr - center);
        br = center + scale * (br - center);
        bl = center + scale * (bl - center);
    }

    mesh->vertices_ = {tl, tr, br, bl};
    mesh->triangles_ = {{0, 2, 1}, {0, 3, 2}};

    mesh->triangle_uvs_ = {
        {0.0, 0.0}, {1.0, 1.0}, {1.0, 0.0},  // tri 0: tl, br, tr
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0},  // tri 1: tl, bl, br
    };

    auto img = std::make_shared<o3d::geometry::Image>();
    img->Prepare(1, 1, 3, 1);
    mesh->textures_.push_back(*img);
    mesh->triangle_material_ids_ = {0, 0};

    return mesh;
}

// ── Update the texture on an image plane mesh ─────────────────────────────────

static void updateImagePlaneTexture(o3d::geometry::TriangleMesh& mesh,
                                    const cv::Mat& rgb)
{
    auto& tex = mesh.textures_[0];
    tex.Prepare(rgb.cols, rgb.rows, 3, 1);
    memcpy(tex.data_.data(), rgb.data, rgb.cols * rgb.rows * 3);
}

// ── Create a camera frustum as a TriangleMesh (thick lines via cylinders) ─────

static std::shared_ptr<o3d::geometry::TriangleMesh>
createCameraFrustum(const double T[4][4],
                    double fx, double fy, double cx, double cy,
                    int imgW, int imgH, double depth,
                    const Eigen::Vector3d& color,
                    double radius = 0.002)
{
    auto origin = transformPt(T, 0, 0, 0);
    auto tl = unprojectCorner(0,    0,    fx, fy, cx, cy, depth, T);
    auto tr = unprojectCorner(imgW, 0,    fx, fy, cx, cy, depth, T);
    auto br = unprojectCorner(imgW, imgH, fx, fy, cx, cy, depth, T);
    auto bl = unprojectCorner(0,    imgH, fx, fy, cx, cy, depth, T);

    std::vector<Eigen::Vector3d> pts = {origin, tl, tr, br, bl};
    // 4 lines from origin to corners + 4 connecting corners
    int edges[][2] = {{0,1},{0,2},{0,3},{0,4}, {1,2},{2,3},{3,4},{4,1}};

    auto combined = std::make_shared<o3d::geometry::TriangleMesh>();
    for (auto& e : edges) {
        auto cyl = o3d::geometry::TriangleMesh::CreateCylinder(radius, (pts[e[1]] - pts[e[0]]).norm(), 6, 1);

        // Orient cylinder from pts[e[0]] to pts[e[1]]
        Eigen::Vector3d dir = (pts[e[1]] - pts[e[0]]).normalized();
        Eigen::Vector3d up(0, 0, 1);  // cylinder default axis
        Eigen::Vector3d mid = (pts[e[0]] + pts[e[1]]) * 0.5;

        // Rotation from Z-axis to dir
        Eigen::Vector3d axis = up.cross(dir);
        double angle = std::acos(std::clamp(up.dot(dir), -1.0, 1.0));
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        if (axis.norm() > 1e-6) {
            axis.normalize();
            Eigen::AngleAxisd rot(angle, axis);
            transform.block<3,3>(0,0) = rot.toRotationMatrix();
        } else if (up.dot(dir) < 0) {
            // 180-degree flip
            transform.block<3,3>(0,0) = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1,0,0)).toRotationMatrix();
        }
        transform.block<3,1>(0,3) = mid;
        cyl->Transform(transform);
        *combined += *cyl;
    }

    // Add a small sphere at the camera origin
    auto sphere = o3d::geometry::TriangleMesh::CreateSphere(radius * 3, 8);
    sphere->Translate(origin);
    *combined += *sphere;

    combined->PaintUniformColor(color);
    combined->ComputeVertexNormals();

    return combined;
}

// ── Create a 3D text label at a camera position ──────────────────────────────

static std::shared_ptr<o3d::geometry::TriangleMesh>
createCameraLabel(const double T[4][4], const std::string& text,
                  const Eigen::Vector3d& color, double scale = 0.02)
{
    // Create text mesh using tensor API, then convert to legacy
    auto tmesh = o3d::t::geometry::TriangleMesh::CreateText(text, 1.0);
    auto legacy = tmesh.ToLegacy();

    // Scale and position the text
    // Text is created in XY plane; we need to transform it to the camera position
    // First scale it down
    Eigen::Matrix4d S = Eigen::Matrix4d::Identity();
    S(0,0) = scale; S(1,1) = scale; S(2,2) = scale;

    // Build the camera's 4x4 transform as Eigen matrix
    Eigen::Matrix4d camT = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            camT(r, c) = T[r][c];

    // Place text slightly above the camera (offset in camera's local Y direction)
    Eigen::Matrix4d offset = Eigen::Matrix4d::Identity();
    offset(1, 3) = -1.5;  // shift up in local frame (before scaling)

    legacy.Transform(camT * S * offset);

    // Paint uniform color
    legacy.PaintUniformColor(color);

    return std::make_shared<o3d::geometry::TriangleMesh>(std::move(legacy));
}

// ── Convert T265 pose to 4×4 Eigen matrix ─────────────────────────────────────

static Eigen::Matrix4d poseToMatrix(const rs2_pose& pose)
{
    Eigen::Quaterniond q(pose.rotation.w, pose.rotation.x,
                         pose.rotation.y, pose.rotation.z);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = q.toRotationMatrix();
    T(0,3) = pose.translation.x;
    T(1,3) = pose.translation.y;
    T(2,3) = pose.translation.z;
    return T;
}

// ── Apply 4×4 Eigen transform to a double[4][4] → result double[4][4] ────────

static void multiplyT44(const Eigen::Matrix4d& A, const double B[4][4], double C[4][4])
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            C[r][c] = 0;
            for (int k = 0; k < 4; ++k)
                C[r][c] += A(r, k) * B[k][c];
        }
}

// ── Load camera intrinsics (fx, fy, cx, cy) from YAML ────────────────────────

struct CamIntrinsics { double fx, fy, cx, cy; int w, h; };

static bool loadIntrinsics(const std::string& path, CamIntrinsics& ci)
{
    cv::FileStorage fss(path, cv::FileStorage::READ);
    if (!fss.isOpened()) return false;
    cv::Mat K;
    fss["K"] >> K;
    if (K.empty()) return false;
    K.convertTo(K, CV_64F);
    ci.fx = K.at<double>(0, 0);
    ci.fy = K.at<double>(1, 1);
    ci.cx = K.at<double>(0, 2);
    ci.cy = K.at<double>(1, 2);
    ci.w  = (int)fss["image_width"].real();
    ci.h  = (int)fss["image_height"].real();
    return true;
}

// ── Unproject one camera's aligned depth+color → append to cloud ──────────────

static void appendCam(RsCam& cam, float minDep, float maxDep, int stride,
                      std::vector<Eigen::Vector3d>& pts,
                      std::vector<Eigen::Vector3d>& cols)
{
    rs2::frameset frames;
    try {
        frames = cam.pipe.wait_for_frames(200);   // 200ms timeout
    } catch (...) {
        std::cerr << "[WARN] " << cam.name << " frame timeout\n";
        return;
    }
    frames = cam.align_to_color.process(frames);

    auto df = frames.get_depth_frame();
    auto cf = frames.get_color_frame();
    if (!df || !cf) return;

    int W = df.get_width();
    int H = df.get_height();
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(df.get_data());
    const uint8_t*  rgb = reinterpret_cast<const uint8_t*> (cf.get_data());

    const double* T = &cam.T[0][0];

    for (int v = 0; v < H; v += stride) {
        for (int u = 0; u < W; u += stride) {
            uint16_t d = raw[v * W + u];
            if (d == 0) continue;
            float dm = d * cam.depth_scale;
            if (dm < minDep || dm > maxDep) continue;

            float pixel[2] = {(float)u, (float)v};
            float pt[3];
            rs2_deproject_pixel_to_point(pt, &cam.color_intr, pixel, dm);

            // Transform into cam1 frame
            double xw = T[0]*pt[0] + T[1]*pt[1] + T[2]*pt[2]  + T[3];
            double yw = T[4]*pt[0] + T[5]*pt[1] + T[6]*pt[2]  + T[7];
            double zw = T[8]*pt[0] + T[9]*pt[1] + T[10]*pt[2] + T[11];

            pts.emplace_back(xw, yw, zw);

            // Use actual RGB color from the aligned color frame
            int ci = (v * W + u) * 3;
            cols.emplace_back(rgb[ci] / 255.0, rgb[ci+1] / 255.0, rgb[ci+2] / 255.0);
        }
    }
}

// ── CLI ───────────────────────────────────────────────────────────────────────

static void printHelp(const char* prog)
{
    std::cout
        << "Real-time multi-camera point cloud fusion viewer.\n\n"
        << "Usage: " << prog << " [options]\n\n"
        << "Cameras:\n"
        << "  --cameras <list>      Comma-separated camera list (default: cam1,cam2,cam4,cam5)\n"
        << "  --config <path>       Camera config YAML (default: configs/cameras.yaml)\n"
        << "  --transforms <path>   Extrinsic transforms (default: results/extrinsics/all_to_cam1_extra.yaml)\n\n"
        << "Depth:\n"
        << "  --min_depth <m>       Minimum depth threshold (default: 0.25, L515 min)\n"
        << "  --max_depth <m>       Maximum depth threshold (default: 4.0)\n"
        << "  --stride <px>         Pixel step — higher = fewer points, faster (default: 4)\n"
        << "  --voxel <m>           Voxel downsample size, 0 = off (default: 0)\n\n"
        << "Stream:\n"
        << "  --res <WxH>           Stream resolution, e.g. 640x480 (default: auto)\n"
        << "  --fps <N>             Stream framerate (default: 15)\n\n"
        << "ZED image plane:\n"
        << "  --no_zed              Skip ZED camera RGB plane\n"
        << "  --zed_dev <N>         ZED V4L2 device index (default: from config)\n"
        << "  --zed_scale <f>       ZED texture resolution scale (default: 0.5)\n"
        << "  --plane_depth <m>     Distance of image plane from ZED camera (default: 1.5)\n"
        << "  --plane_scale <f>     Size scale of image plane, 1.0 = full (default: 0.42)\n\n"
        << "Display:\n"
        << "  --no_frustums         Hide camera frustum wireframes\n"
        << "  --t265                Enable T265 tracking (coordinate axes + trajectory trace)\n\n"
        << "Controls:\n"
        << "  Left drag   — rotate       Right drag/scroll — zoom\n"
        << "  Ctrl+drag   — pan          Q / close window  — stop\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    std::string configPath     = "configs/cameras.yaml";
    std::string transformsPath = "results/extrinsics/all_to_cam1_extra.yaml";
    std::string camerasArg     = "cam1,cam2,cam4,cam5";   // default: all
    float minDep  = 0.25f;
    float maxDep  = 4.0f;
    int   stride  = 4;      // default 4 — lighter workload
    double voxel  = 0.0;
    bool  noZed   = false;
    bool  noFrustums = false;
    bool  useT265 = false;
    int   zedDevOverride = -1;
    double zedScale = 0.5;
    double planeDepth = 1.5;
    double planeScale = 0.42;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { printHelp(argv[0]); return 0; }
        else if (a == "--config"      && i+1 < argc) configPath     = argv[++i];
        else if (a == "--transforms"  && i+1 < argc) transformsPath = argv[++i];
        else if (a == "--cameras"     && i+1 < argc) camerasArg     = argv[++i];
        else if (a == "--min_depth"   && i+1 < argc) minDep         = std::stof(argv[++i]);
        else if (a == "--max_depth"   && i+1 < argc) maxDep         = std::stof(argv[++i]);
        else if (a == "--stride"      && i+1 < argc) stride         = std::stoi(argv[++i]);
        else if (a == "--voxel"       && i+1 < argc) voxel          = std::stod(argv[++i]);
        else if (a == "--fps"         && i+1 < argc) streamFps      = std::stoi(argv[++i]);
        else if (a == "--zed_dev"     && i+1 < argc) zedDevOverride = std::stoi(argv[++i]);
        else if (a == "--zed_scale"   && i+1 < argc) zedScale       = std::stod(argv[++i]);
        else if (a == "--plane_depth" && i+1 < argc) planeDepth     = std::stod(argv[++i]);
        else if (a == "--plane_scale" && i+1 < argc) planeScale     = std::stod(argv[++i]);
        else if (a == "--no_zed")       noZed = true;
        else if (a == "--no_frustums")  noFrustums = true;
        else if (a == "--t265")         useT265 = true;
        else if (a == "--res"         && i+1 < argc) {
            std::string r = argv[++i];
            auto x = r.find('x');
            if (x != std::string::npos) {
                streamW = std::stoi(r.substr(0, x));
                streamH = std::stoi(r.substr(x+1));
            }
        }
    }

    // Parse comma-separated camera list
    std::set<std::string> wantCams;
    {
        std::istringstream ss(camerasArg);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) wantCams.insert(tok);
    }

    // Load camera configs
    auto configs = loadCameraConfigs(configPath);
    if (configs.empty()) {
        std::cerr << "[ERROR] No camera configs loaded from " << configPath << "\n";
        return 1;
    }

    // Load transforms (all 6 cameras from all_to_cam1_extra.yaml)
    double T_1_1[4][4], T_1_2[4][4], T_1_3L[4][4], T_1_4[4][4], T_1_5[4][4], T_1_3R[4][4];
    if (!loadT44(transformsPath, "T_1_cam1", T_1_1) ||
        !loadT44(transformsPath, "T_1_cam2", T_1_2) ||
        !loadT44(transformsPath, "T_1_cam3_left", T_1_3L) ||
        !loadT44(transformsPath, "T_1_cam4", T_1_4) ||
        !loadT44(transformsPath, "T_1_cam5", T_1_5) ||
        !loadT44(transformsPath, "T_1_cam3_right", T_1_3R)) {
        return 1;
    }

    // Per-camera display colors (RGB 0-1)
    const Eigen::Vector3d kRed     = {1.0, 0.2, 0.2};   // cam1
    const Eigen::Vector3d kGreen   = {0.2, 1.0, 0.2};   // cam2
    const Eigen::Vector3d kCyan    = {0.2, 1.0, 1.0};   // cam3_left
    const Eigen::Vector3d kBlue    = {0.2, 0.2, 1.0};   // cam3_right
    const Eigen::Vector3d kYellow  = {1.0, 1.0, 0.2};   // cam4
    const Eigen::Vector3d kMagenta = {1.0, 0.2, 1.0};   // cam5

    // Enumerate connected RealSense devices once (avoids interfering contexts later)
    std::set<std::string> connected;
    {
        rs2::context ctx;
        for (auto&& dev : ctx.query_devices())
            connected.insert(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    }
    std::cout << "[INFO] Connected serials:";
    for (auto& s : connected) std::cout << " " << s;
    std::cout << "\n";

    // Try opening a camera with a 6-second timeout.
    // Uses a detached thread so pipe.start() hanging doesn't block the caller.
    auto tryOpen = [&](const std::string& name, const std::string& serial,
                       const Eigen::Vector3d& clr) -> std::unique_ptr<RsCam> {
        if (!connected.count(serial)) {
            std::cerr << "[SKIP] " << name << " (" << serial << ") not found.\n";
            return nullptr;
        }
        std::cout << "[INFO] Opening " << name << "..." << std::flush;

        // Shared state between caller and detached thread
        struct Slot {
            std::unique_ptr<RsCam> cam;
            bool done = false;
            std::mutex mtx;
            std::condition_variable cv;
        };
        auto slot = std::make_shared<Slot>();

        std::thread([slot, name, serial, clr]() mutable {
            auto c = openCam(name, serial, clr);
            std::lock_guard<std::mutex> lk(slot->mtx);
            slot->cam  = std::move(c);
            slot->done = true;
            slot->cv.notify_one();
        }).detach();   // detach immediately — won't block on destructor

        std::unique_lock<std::mutex> lk(slot->mtx);
        bool ok = slot->cv.wait_for(lk, std::chrono::seconds(6),
                                    [&]{ return slot->done; });
        if (!ok) {
            std::cerr << "\n[SKIP] " << name << " timed out (USB bandwidth?). Continuing.\n";
            return nullptr;   // slot kept alive by detached thread's shared_ptr copy
        }
        return std::move(slot->cam);
    };

    // Open only the cameras in wantCams
    auto maybeOpen = [&](const std::string& name, const Eigen::Vector3d& clr,
                         double T[4][4]) -> std::unique_ptr<RsCam> {
        if (!wantCams.count(name)) return nullptr;
        auto cam = tryOpen(name, configs.at(name).serial, clr);
        if (cam) memcpy(cam->T, T, sizeof(double[4][4]));
        return cam;
    };

    std::cout << "[INFO] Opening cameras: " << camerasArg << "\n";
    auto cam1 = maybeOpen("cam1", kRed,     T_1_1);
    auto cam2 = maybeOpen("cam2", kGreen,   T_1_2);
    auto cam4 = maybeOpen("cam4", kYellow,  T_1_4);
    auto cam5 = maybeOpen("cam5", kMagenta, T_1_5);

    int activeCams = (cam1 ? 1 : 0) + (cam2 ? 1 : 0) + (cam4 ? 1 : 0) + (cam5 ? 1 : 0);
    if (activeCams == 0) {
        std::cerr << "[ERROR] No cameras available.\n";
        return 1;
    }
    std::cout << "[INFO] " << activeCams << " camera(s) active.\n";

    std::cout << "[INFO] stride=" << stride
              << "  depth=[" << minDep << ", " << maxDep << "]m"
              << "  voxel=" << (voxel > 0 ? std::to_string(voxel) + "m" : "off")
              << "\n";

    // ── Load intrinsics for all 6 cameras (for frustums + labels) ──────────────

    struct FrustumInfo {
        CamIntrinsics intr;
        double T[4][4];
        Eigen::Vector3d color;
        std::string name;
        std::string label;  // model name for 3D text
    };
    std::vector<FrustumInfo> frustumInfos;
    {
        // path, T ptr, color, short name, label
        struct E { std::string p; double(*T)[4]; Eigen::Vector3d c; std::string n; std::string l; };
        std::vector<E> entries = {
            {"results/intrinsics/cam1.yaml",       T_1_1,  kRed,     "cam1", "L515-1"},
            {"results/intrinsics/cam2.yaml",       T_1_2,  kGreen,   "cam2", "L515-2"},
            {"results/intrinsics/cam3_left.yaml",  T_1_3L, kCyan,    "cam3L", "ZED-L"},
            {"results/intrinsics/cam3_right.yaml", T_1_3R, kBlue,    "cam3R", "ZED-R"},
            {"results/intrinsics/cam4.yaml",       T_1_4,  kYellow,  "cam4", "D405-4"},
            {"results/intrinsics/cam5.yaml",       T_1_5,  kMagenta, "cam5", "D405-5"},
        };
        for (auto& e : entries) {
            FrustumInfo fi;
            if (loadIntrinsics(e.p, fi.intr)) {
                memcpy(fi.T, e.T, sizeof(double[4][4]));
                fi.color = e.c;
                fi.name = e.n;
                fi.label = e.l;
                frustumInfos.push_back(fi);
            } else {
                std::cerr << "[WARN] Cannot load intrinsics: " << e.p << "\n";
            }
        }
    }

    // ── Create camera frustum wireframes + labels ─────────────────────────────

    std::vector<std::shared_ptr<o3d::geometry::TriangleMesh>> frustums;
    if (!noFrustums) {
        constexpr double kFrustumDepth = 0.06;
        for (auto& fi : frustumInfos) {
            auto mesh = createCameraFrustum(fi.T,
                fi.intr.fx, fi.intr.fy, fi.intr.cx, fi.intr.cy,
                fi.intr.w, fi.intr.h, kFrustumDepth, fi.color, 0.0015);
            frustums.push_back(mesh);
            std::cout << "[INFO] Frustum: " << fi.name << " (" << fi.label << ")\n";
        }
    }

    // ── Open ZED camera via V4L2 ─────────────────────────────────────────────

    cv::VideoCapture zedCap;
    std::shared_ptr<o3d::geometry::TriangleMesh> zedMesh;

    if (!noZed) {
        int zedDev = zedDevOverride >= 0 ? zedDevOverride :
                     (configs.count("cam3") ? configs.at("cam3").dev : -1);
        if (zedDev >= 0) {
            std::cout << "[INFO] Opening ZED on /dev/video" << zedDev << "..." << std::flush;
            zedCap.open(zedDev, cv::CAP_V4L2);
            if (zedCap.isOpened()) {
                zedCap.set(cv::CAP_PROP_FRAME_WIDTH, 2560);
                zedCap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
                // Warm-up
                cv::Mat tmp;
                for (int i = 0; i < 5; ++i) zedCap.read(tmp);
                std::cout << " ok\n";

                // Single plane showing left camera image
                CamIntrinsics ciL;
                loadIntrinsics("results/intrinsics/cam3_left.yaml", ciL);

                zedMesh = createImagePlane(T_1_3L,
                    ciL.fx, ciL.fy, ciL.cx, ciL.cy,
                    ciL.w, ciL.h, planeDepth, planeScale);
            } else {
                std::cerr << " FAILED\n";
            }
        } else {
            std::cout << "[INFO] No ZED device configured, skipping.\n";
        }
    }

    // ── Open T265 for rig pose tracking ─────────────────────────────────────

    rs2::pipeline t265pipe;
    bool t265active = false;

    if (useT265) {
        std::cout << "[INFO] Starting T265 (firmware load may take a few seconds)..." << std::flush;
        try {
            rs2::config t265cfg;
            t265cfg.enable_device("905312111793");
            t265cfg.enable_stream(RS2_STREAM_POSE, RS2_FORMAT_6DOF);
            t265pipe.start(t265cfg);
            for (int i = 0; i < 60; ++i) t265pipe.wait_for_frames(1000);
            t265active = true;
            std::cout << " ok\n";
        } catch (const std::exception& e) {
            std::cerr << " FAILED: " << e.what() << "\n";
        }
    }

    // ── Open3D visualizer setup ───────────────────────────────────────────────

    auto cloud = std::make_shared<o3d::geometry::PointCloud>();
    auto frame = o3d::geometry::TriangleMesh::CreateCoordinateFrame(0.1);

    o3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Point Cloud Stream — " + camerasArg, 1280, 720);
    vis.AddGeometry(cloud);
    vis.AddGeometry(frame);

    // Add frustums
    for (auto& f : frustums) vis.AddGeometry(f);

    // Add ZED stereo image plane
    if (zedMesh) vis.AddGeometry(zedMesh);

    // T265 trajectory trace + moving axes
    auto t265trace = std::make_shared<o3d::geometry::LineSet>();
    auto t265axes  = std::make_shared<o3d::geometry::TriangleMesh>();
    bool t265trace_added = false;
    if (t265active) {
        // Create initial small coordinate frame for T265 pose
        *t265axes = *o3d::geometry::TriangleMesh::CreateCoordinateFrame(0.05);
        vis.AddGeometry(t265axes);
        // Don't add trace yet — wait until it has at least one segment
    }

    // Set a reasonable initial viewpoint (look from +Z toward origin)
    auto& vc = vis.GetViewControl();
    vc.SetZoom(0.6);

    std::cout << "[INFO] Streaming — close window or press Q to stop.\n\n";

    int frameIdx = 0;
    bool viewReset = false;
    Eigen::Matrix4d t265_prev_pose = Eigen::Matrix4d::Identity();

    // Store initial ZED plane vertices for T265-driven rotation
    auto t265_axes_base = o3d::geometry::TriangleMesh::CreateCoordinateFrame(0.05);
    Eigen::Vector3d t265_last_trace_pt(0, 0, 0);
    bool t265_has_trace = false;

    while (vis.PollEvents()) {
        // ── Update T265 pose visualization ───────────────────────────────
        if (t265active) {
            rs2::frameset t265frames;
            try {
                t265frames = t265pipe.wait_for_frames(10);
                auto pf = t265frames.first_or_default(RS2_STREAM_POSE);
                if (pf) {
                    auto pose = pf.as<rs2::pose_frame>().get_pose_data();
                    Eigen::Matrix4d T = poseToMatrix(pose);

                    // Update moving coordinate frame
                    *t265axes = *t265_axes_base;
                    t265axes->Transform(T);
                    vis.UpdateGeometry(t265axes);

                    // Add to trajectory trace (every few cm to avoid clutter)
                    Eigen::Vector3d pos(T(0,3), T(1,3), T(2,3));
                    if (!t265_has_trace || (pos - t265_last_trace_pt).norm() > 0.005) {
                        if (t265_has_trace) {
                            size_t n = t265trace->points_.size();
                            t265trace->points_.push_back(pos);
                            t265trace->lines_.push_back({(int)n - 1, (int)n});
                            // Color by confidence: green=high, yellow=medium, red=low
                            double g = std::min(1.0, pose.tracker_confidence / 3.0);
                            t265trace->colors_.push_back({1.0 - g, g, 0.2});
                        } else {
                            t265trace->points_.push_back(pos);
                        }
                        t265_last_trace_pt = pos;
                        t265_has_trace = true;
                        if (!t265trace_added && !t265trace->lines_.empty()) {
                            vis.AddGeometry(t265trace);
                            t265trace_added = true;
                        } else if (t265trace_added) {
                            vis.UpdateGeometry(t265trace);
                        }
                    }

                    // T265 shown as independent axes + trace only

                    // Print pose info periodically
                    if (frameIdx % 90 == 0) {
                        auto rv = Eigen::AngleAxisd(T.block<3,3>(0,0));
                        std::cout << "\r  T265: t=[" << std::fixed << std::setprecision(3)
                                  << pos.x() << "," << pos.y() << "," << pos.z()
                                  << "] rot=" << std::setprecision(1)
                                  << rv.angle() * 180.0 / M_PI << "deg"
                                  << " conf=" << pose.tracker_confidence
                                  << "          " << std::flush;
                    }
                }
            } catch (...) {}
        }

        std::vector<Eigen::Vector3d> pts, cols;
        pts.reserve(4 * (640 / stride) * (480 / stride));
        cols.reserve(pts.capacity());

        if (cam1) appendCam(*cam1, minDep, maxDep, stride, pts, cols);
        if (cam2) appendCam(*cam2, minDep, maxDep, stride, pts, cols);
        if (cam4) appendCam(*cam4, minDep, maxDep, stride, pts, cols);
        if (cam5) appendCam(*cam5, minDep, maxDep, stride, pts, cols);

        cloud->points_ = std::move(pts);
        cloud->colors_ = std::move(cols);

        if (voxel > 0.0) {
            auto down = cloud->VoxelDownSample(voxel);
            cloud->points_ = down->points_;
            cloud->colors_ = down->colors_;
        }

        vis.UpdateGeometry(cloud);

        // Update ZED left camera image plane
        if (zedCap.isOpened() && zedMesh) {
            cv::Mat sbs;
            if (zedCap.read(sbs) && sbs.cols >= 2560 && sbs.rows >= 720) {
                // Extract left half, convert BGR → RGB
                cv::Mat leftBGR = sbs(cv::Rect(0, 0, 1280, 720));
                cv::Mat leftRGB;
                cv::cvtColor(leftBGR, leftRGB, cv::COLOR_BGR2RGB);

                if (zedScale < 1.0)
                    cv::resize(leftRGB, leftRGB, cv::Size(), zedScale, zedScale);

                updateImagePlaneTexture(*zedMesh, leftRGB);
                vis.UpdateGeometry(zedMesh);
            }
        }

        // Reset viewpoint once after the first real frame so the camera
        // fits to the actual point cloud (not the initial empty bounding box)
        if (!viewReset && !cloud->points_.empty()) {
            vis.ResetViewPoint(true);
            viewReset = true;
        }

        vis.UpdateRender();

        if (++frameIdx % 30 == 0)
            std::cout << "\r  pts=" << cloud->points_.size()
                      << "  frame=" << frameIdx << std::flush;
    }

    std::cout << "\n[INFO] Window closed — stopping cameras.\n";
    vis.DestroyVisualizerWindow();

    if (t265active) t265pipe.stop();
    if (zedCap.isOpened()) zedCap.release();
    zedMesh.reset();
    if (cam1) cam1->pipe.stop();
    if (cam2) cam2->pipe.stop();
    if (cam4) cam4->pipe.stop();
    if (cam5) cam5->pipe.stop();

    return 0;
}
