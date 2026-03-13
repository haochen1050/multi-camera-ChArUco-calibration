// =============================================================================
//  calibrate_extrinsics_pair.cpp
//  Pairwise extrinsic calibration between two cameras using ChArUco.
//  Always reads intrinsics from YAML — link against no camera SDK.
//
//  Usage:
//    calibrate_extrinsics_pair \
//        --images_a <dir>  --intr_a <yaml> \
//        --images_b <dir>  --intr_b <yaml> \
//        --name <camA_camB> [--out <results/extrinsics>]
//
//  Output:
//    results/extrinsics/<name>.yaml  (T_a_b 4×4, R, T, E, F, rms)
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

#include "calib_common.hpp"

namespace fs = std::filesystem;

static bool loadIntrinsics(const std::string& path,
                            cv::Mat& K, cv::Mat& D, cv::Size& sz)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) { std::cerr << "[ERROR] Cannot open: " << path << "\n"; return false; }
    fs["K"] >> K;
    fs["D"] >> D;
    int w = 0, h = 0;
    fs["image_width"]  >> w;
    fs["image_height"] >> h;
    sz = cv::Size(w, h);
    K.convertTo(K, CV_64F);
    D.convertTo(D, CV_64F);
    return !K.empty() && !D.empty() && w > 0 && h > 0;
}

// Collect per-image detections from a sorted directory listing.
// Returns sorted list of (ids, corners) per image — empty entry if no detection.
static void collectDetections(
    const std::string& dir,
    const cv::Ptr<cv::aruco::CharucoBoard>& board,
    const cv::Mat& K, const cv::Mat& D,
    std::vector<std::vector<int>>&         allIds,
    std::vector<std::vector<cv::Point2f>>& allCorners,
    std::vector<std::string>&              names)
{
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(dir)) {
        std::string ext = e.path().extension().string();
        if (ext == ".png" || ext == ".jpg") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());

    for (auto& p : files) {
        cv::Mat img = cv::imread(p.string());
        std::vector<int>         ids;
        std::vector<cv::Point2f> corners;
        if (!img.empty())
            detectCharuco(img, board, ids, corners, 1, K, D);
        allIds.push_back(ids);
        allCorners.push_back(corners);
        names.push_back(p.filename().string());
    }
}

static void printHelp(const char* prog)
{
    std::cout
        << "Usage: " << prog << "\n"
        << "  --images_a <dir>   Image folder for camera A\n"
        << "  --intr_a   <yaml>  Intrinsics YAML for camera A\n"
        << "  --images_b <dir>   Image folder for camera B\n"
        << "  --intr_b   <yaml>  Intrinsics YAML for camera B\n"
        << "  --name     <str>   Output file stem  (e.g. cam1_cam2)\n"
        << "  --out      <dir>   Output directory  (default: results/extrinsics)\n";
}

int main(int argc, char** argv)
{
    std::string dirA, intrA, dirB, intrB, name;
    std::string outDir = "results/extrinsics";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { printHelp(argv[0]); return 0; }
        else if (a == "--images_a" && i+1 < argc) dirA   = argv[++i];
        else if (a == "--intr_a"   && i+1 < argc) intrA  = argv[++i];
        else if (a == "--images_b" && i+1 < argc) dirB   = argv[++i];
        else if (a == "--intr_b"   && i+1 < argc) intrB  = argv[++i];
        else if (a == "--name"     && i+1 < argc) name   = argv[++i];
        else if (a == "--out"      && i+1 < argc) outDir = argv[++i];
    }

    if (dirA.empty() || intrA.empty() || dirB.empty() || intrB.empty() || name.empty()) {
        printHelp(argv[0]);
        return 1;
    }

    cv::Mat KA, DA, KB, DB;
    cv::Size szA, szB;
    if (!loadIntrinsics(intrA, KA, DA, szA)) return 1;
    if (!loadIntrinsics(intrB, KB, DB, szB)) return 1;

    auto board = buildCharucoBoard();

    std::vector<std::vector<int>>         idsA, idsB;
    std::vector<std::vector<cv::Point2f>> cornersA, cornersB;
    std::vector<std::string>              namesA, namesB;

    collectDetections(dirA, board, KA, DA, idsA, cornersA, namesA);
    collectDetections(dirB, board, KB, DB, idsB, cornersB, namesB);

    if (idsA.size() != idsB.size()) {
        std::cerr << "[ERROR] Image count mismatch: A=" << idsA.size()
                  << " B=" << idsB.size() << ". Sort by filename.\n";
        return 1;
    }

    // Build matched object/image point sets (common corner IDs per frame pair)
    std::vector<std::vector<cv::Point3f>> objPtsAll;
    std::vector<std::vector<cv::Point2f>> imgPtsA_all, imgPtsB_all;

    int nUsed = 0;
    for (size_t f = 0; f < idsA.size(); ++f) {
        if (idsA[f].empty() || idsB[f].empty()) continue;

        // Find common corner IDs
        std::vector<int>         commonIds;
        std::vector<cv::Point2f> matchA, matchB;

        for (size_t ia = 0; ia < idsA[f].size(); ++ia) {
            int id = idsA[f][ia];
            auto itB = std::find(idsB[f].begin(), idsB[f].end(), id);
            if (itB != idsB[f].end()) {
                size_t ib = std::distance(idsB[f].begin(), itB);
                commonIds.push_back(id);
                matchA.push_back(cornersA[f][ia]);
                matchB.push_back(cornersB[f][ib]);
            }
        }
        if (static_cast<int>(commonIds.size()) < 15) continue;

        // Get 3D object points for common IDs
        std::vector<cv::Point3f> obj;
        std::vector<cv::Point2f> imgA_ordered, imgB_ordered;
        for (size_t k = 0; k < commonIds.size(); ++k) {
            int id = commonIds[k];
            if (id >= 0 && id < static_cast<int>(board->chessboardCorners.size())) {
                obj.push_back(board->chessboardCorners[id]);
                imgA_ordered.push_back(matchA[k]);
                imgB_ordered.push_back(matchB[k]);
            }
        }
        if (static_cast<int>(obj.size()) < 15) continue;

        objPtsAll.push_back(obj);
        imgPtsA_all.push_back(imgA_ordered);
        imgPtsB_all.push_back(imgB_ordered);
        ++nUsed;
    }

    std::cout << "[INFO] Valid frame pairs for stereoCalibrate: " << nUsed << "\n";
    if (nUsed < 5) { std::cerr << "[ERROR] Not enough valid frames.\n"; return 1; }

    cv::Mat R, T, E, F;
    auto criteria = cv::TermCriteria(
        cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 200, 1e-7);

    double rms = cv::stereoCalibrate(
        objPtsAll, imgPtsA_all, imgPtsB_all,
        KA, DA, KB, DB,
        szA, R, T, E, F,
        cv::CALIB_FIX_INTRINSIC, criteria);

    std::cout << "  Stereo RMS: " << rms << " px\n";

    // stereoCalibrate gives R,T mapping camA→camB (p_B = R*p_A + T).
    // Our convention: T_a_b maps FROM b INTO a (p_a = T_a_b * p_b).
    // So T_a_b = inv([R T; 0 1]).
    cv::Mat T44_raw = makeT44(R, T);
    cv::Mat T44;
    cv::invert(T44_raw, T44);
    printTransform("T_" + name + ":", T44);

    fs::create_directories(outDir);
    std::string outPath = outDir + "/" + name + ".yaml";
    {
        cv::FileStorage fs(outPath, cv::FileStorage::WRITE);
        fs << "pair"  << name;
        fs << "T_a_b" << T44;
        fs << "R"     << R;
        fs << "T"     << T;
        fs << "E"     << E;
        fs << "F"     << F;
        fs << "rms"   << rms;
        fs.release();
    }
    std::cout << "[SAVE] " << outPath << "\n";
    return 0;
}
