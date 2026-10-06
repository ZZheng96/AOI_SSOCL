#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// ============================================================
// compute_diff: BGR三通道差分 + 阈值 + 形态学 + 小区域过滤
// ============================================================
py::tuple compute_diff_fast(
    py::array_t<uint8_t> template_crop,
    py::array_t<uint8_t> aligned_inspect,
    py::dict info,
    py::dict params)
{
    auto t_buf = template_crop.request();
    auto i_buf = aligned_inspect.request();
    if (t_buf.ndim != 3 || i_buf.ndim != 3) {
        throw std::runtime_error("compute_diff_fast: expected 3-channel images");
    }

    cv::Mat t_img(t_buf.shape[0], t_buf.shape[1], CV_8UC3, t_buf.ptr);
    cv::Mat i_img(i_buf.shape[0], i_buf.shape[1], CV_8UC3, i_buf.ptr);

    // 直接使用已对齐的crop, 无需warpAffine
    cv::Mat aligned = i_img;

    // 三通道差分取max
    cv::Mat diff(t_img.size(), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < t_img.rows; y++) {
        const uint8_t* tp = t_img.ptr<uint8_t>(y);
        const uint8_t* ap = aligned.ptr<uint8_t>(y);
        uint8_t* dp = diff.ptr<uint8_t>(y);
        for (int x = 0; x < t_img.cols; x++) {
            int b = std::abs((int)tp[0] - (int)ap[0]);
            int g = std::abs((int)tp[1] - (int)ap[1]);
            int r = std::abs((int)tp[2] - (int)ap[2]);
            dp[x] = (uint8_t)std::max({b, g, r});
            tp += 3; ap += 3;
        }
    }

    // 阈值
    int thresh = params.contains("diff_thresh") ? params["diff_thresh"].cast<int>() : 80;
    cv::Mat diff_mask;
    cv::threshold(diff, diff_mask, thresh, 255, cv::THRESH_BINARY);

    // 形态学去噪 (3x3)
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::morphologyEx(diff_mask, diff_mask, cv::MORPH_OPEN, k);
    cv::morphologyEx(diff_mask, diff_mask, cv::MORPH_CLOSE, k);

    // 小区域过滤
    int min_area = params.contains("diff_min_area") ? params["diff_min_area"].cast<int>() : 10;
    if (min_area > 0) {
        cv::Mat labels, stats, centroids;
        int n = cv::connectedComponentsWithStats(diff_mask, labels, stats, centroids, 8);
        diff_mask = cv::Mat::zeros(diff_mask.size(), CV_8UC1);
        for (int j = 1; j < n; j++) {
            if (stats.at<int>(j, cv::CC_STAT_AREA) >= min_area) {
                diff_mask.setTo(255, labels == j);
            }
        }
    }

    // heatmap (默认跳过)
    bool skip_heatmap = params.contains("skip_diff_heatmap") && params["skip_diff_heatmap"].cast<bool>();
    py::object heatmap = py::none();
    if (!skip_heatmap) {
        cv::Mat diff_norm;
        cv::normalize(diff, diff_norm, 0, 255, cv::NORM_MINMAX);
        cv::Mat heat;
        cv::applyColorMap(diff_norm, heat, cv::COLORMAP_JET);
        heatmap = py::array_t<uint8_t>({heat.rows, heat.cols, 3}, heat.data);
    }

    // 返回 aligned_image 用于后续坐标映射
    auto py_aligned = py::array_t<uint8_t>({aligned.rows, aligned.cols, 3}, aligned.data);
    auto py_aligned_info = py::dict(
        py::arg("aligned_inspect") = py_aligned,
        py::arg("template_crop") = info["template_crop"]
    );

    return py::make_tuple(
        py::array_t<uint8_t>({diff_mask.rows, diff_mask.cols}, diff_mask.data),
        heatmap,
        py_aligned_info
    );
}

// ============================================================
// hough裂痕检测: 纯C++实现, 避免Python循环瓶颈
// ============================================================
py::tuple detect_crack_hough_fast(
    py::array_t<uint8_t> gray_roi,
    py::array_t<uint8_t> pad_mask_roi,
    py::dict p)
{
    auto g_buf = gray_roi.request();
    auto m_buf = pad_mask_roi.request();

    cv::Mat gray(g_buf.shape[0], g_buf.shape[1], CV_8UC1, g_buf.ptr);
    cv::Mat pad_mask(m_buf.shape[0], m_buf.shape[1], CV_8UC1, m_buf.ptr);

    int h = gray.rows, w = gray.cols;
    int pad_pixels = cv::countNonZero(pad_mask);
    if (pad_pixels < 50) return py::make_tuple(false, py::list());

    // 1. 暗带mask: GaussianBlur(21x21) + 0.85倍
    cv::Mat gray_blur;
    cv::GaussianBlur(gray, gray_blur, {21, 21}, 0);
    cv::Mat dark_mask;
    cv::compare(gray, gray_blur * 0.85f, dark_mask, cv::CMP_LT);
    dark_mask &= (pad_mask > 0);
    dark_mask.convertTo(dark_mask, CV_8UC1, 255.0);

    // 2. 边缘排除 (15% margin)
    int margin_x = (int)(w * 0.15f);
    int margin_y = (int)(h * 0.15f);
    if (margin_y > 0) {
        dark_mask.rowRange(0, margin_y).setTo(0);
        dark_mask.rowRange(std::max(h - margin_y, 0), h).setTo(0);
    }
    if (margin_x > 0) {
        dark_mask.colRange(0, margin_x).setTo(0);
        dark_mask.colRange(std::max(w - margin_x, 0), w).setTo(0);
    }

    // 3. 形态学闭运算 (1x15竖向 + 15x1水平)
    cv::Mat k_v = cv::getStructuringElement(cv::MORPH_RECT, {1, 15});
    cv::Mat k_h = cv::getStructuringElement(cv::MORPH_RECT, {15, 1});
    cv::morphologyEx(dark_mask, dark_mask, cv::MORPH_CLOSE, k_v);
    cv::morphologyEx(dark_mask, dark_mask, cv::MORPH_CLOSE, k_h);

    // 4. HoughLinesP
    int pad_short = std::min(w, h);
    int hough_thresh = p.contains("crack_hough_threshold") ? p["crack_hough_threshold"].cast<int>() : 15;
    int min_length = p.contains("crack_hough_min_length") ? p["crack_hough_min_length"].cast<int>() : 0;
    if (min_length <= 0) min_length = std::max(pad_short / 3, 5);
    int max_gap = p.contains("crack_hough_max_gap") ? p["crack_hough_max_gap"].cast<int>() : 8;

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(dark_mask, lines, 1, CV_PI / 180.0, hough_thresh, min_length, max_gap);

    if (lines.empty()) return py::make_tuple(false, py::list());

    // 5. 合并共线线段 (简易合并)
    int angle_tol = p.contains("crack_merge_angle_tol") ? p["crack_merge_angle_tol"].cast<int>() : 5;
    std::vector<cv::Vec4i> merged;
    std::vector<bool> used(lines.size(), false);
    for (size_t i = 0; i < lines.size(); i++) {
        if (used[i]) continue;
        cv::Vec4i best = lines[i];
        used[i] = true;
        for (size_t j = i + 1; j < lines.size(); j++) {
            if (used[j]) continue;
            double dx = lines[j][0] - lines[j][2];
            double dy = lines[j][1] - lines[j][3];
            double angle_j = std::abs(std::atan2(dy, dx) * 180.0 / CV_PI);
            double dxb = best[0] - best[2];
            double dyb = best[1] - best[3];
            double angle_b = std::abs(std::atan2(dyb, dxb) * 180.0 / CV_PI);
            double diff = std::abs(angle_j - angle_b);
            if (diff > angle_tol) continue;
            // 端点距离 < 10像素?
            double d1 = std::hypot(lines[j][0] - best[2], lines[j][1] - best[3]);
            double d2 = std::hypot(best[0] - lines[j][2], best[1] - lines[j][3]);
            if (d1 < 10 || d2 < 10) {
                // 合并: 取端点外延
                best[0] = std::min(best[0], lines[j][0]);
                best[1] = std::min(best[1], lines[j][1]);
                best[2] = std::max(best[2], lines[j][2]);
                best[3] = std::max(best[3], lines[j][3]);
                used[j] = true;
            }
        }
        merged.push_back(best);
    }

    // 6-8. 方向/边缘/长度过滤
    bool pad_is_vertical = h > w;
    int edge_margin = std::max(pad_short / 10, 3);
    double min_len_ratio = p.contains("crack_min_length_ratio") ? p["crack_min_length_ratio"].cast<double>() : 0.5;
    std::vector<cv::Vec4i> final_lines;

    for (auto& ln : merged) {
        int x1 = ln[0], y1 = ln[1], x2 = ln[2], y2 = ln[3];
        double line_angle = std::abs(std::atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI);
        if (line_angle > 90) line_angle = 180 - line_angle;
        bool line_is_vertical = line_angle > 45;

        // 6. 方向过滤
        if (pad_is_vertical && line_angle > 5) continue;
        if (!pad_is_vertical && line_angle < 85) continue;

        // 7. 边缘过滤
        bool p1_edge = (x1 < edge_margin || x1 > w - edge_margin ||
                        y1 < edge_margin || y1 > h - edge_margin);
        bool p2_edge = (x2 < edge_margin || x2 > w - edge_margin ||
                        y2 < edge_margin || y2 > h - edge_margin);
        if (p1_edge && p2_edge) continue;

        // 8. 长度过滤
        double length = std::hypot(x2 - x1, y2 - y1);
        if (length > pad_short * min_len_ratio) {
            final_lines.push_back(ln);
        }
    }

    bool has_crack = !final_lines.empty();
    py::list result_lines;
    for (auto& ln : final_lines) {
        result_lines.append(py::make_tuple(ln[0], ln[1], ln[2], ln[3]));
    }

    return py::make_tuple(has_crack, result_lines);
}

// crack检测参数 (从py::dict提取, 避免并行区域内访问Python对象)
struct CrackParams {
    int hough_thresh = 15;
    int min_length = 0;
    int max_gap = 8;
    int angle_tol = 5;
    double min_len_ratio = 0.5;
};

static CrackParams _extract_crack_params(const py::dict& p) {
    CrackParams cp;
    cp.hough_thresh = p.contains("crack_hough_threshold") ? p["crack_hough_threshold"].cast<int>() : 15;
    cp.min_length = p.contains("crack_hough_min_length") ? p["crack_hough_min_length"].cast<int>() : 0;
    cp.max_gap = p.contains("crack_hough_max_gap") ? p["crack_hough_max_gap"].cast<int>() : 8;
    cp.angle_tol = p.contains("crack_merge_angle_tol") ? p["crack_merge_angle_tol"].cast<int>() : 5;
    cp.min_len_ratio = p.contains("crack_min_length_ratio") ? p["crack_min_length_ratio"].cast<double>() : 0.5;
    return cp;
}

// 单个ROI的crack检测 (内部函数, 避免重复代码)
static bool _detect_crack_single(const cv::Mat& gray, const cv::Mat& pad_mask,
                                  const CrackParams& cp) {
    int h = gray.rows, w = gray.cols;
    int pad_pixels = cv::countNonZero(pad_mask);
    if (pad_pixels < 50) return false;

    cv::Mat gray_blur;
    cv::GaussianBlur(gray, gray_blur, {21, 21}, 0);
    cv::Mat dark_mask;
    cv::compare(gray, gray_blur * 0.85f, dark_mask, cv::CMP_LT);
    dark_mask &= (pad_mask > 0);
    dark_mask.convertTo(dark_mask, CV_8UC1, 255.0);

    int margin_x = (int)(w * 0.15f);
    int margin_y = (int)(h * 0.15f);
    if (margin_y > 0) {
        dark_mask.rowRange(0, margin_y).setTo(0);
        dark_mask.rowRange(std::max(h - margin_y, 0), h).setTo(0);
    }
    if (margin_x > 0) {
        dark_mask.colRange(0, margin_x).setTo(0);
        dark_mask.colRange(std::max(w - margin_x, 0), w).setTo(0);
    }

    cv::Mat k_v = cv::getStructuringElement(cv::MORPH_RECT, {1, 15});
    cv::Mat k_h = cv::getStructuringElement(cv::MORPH_RECT, {15, 1});
    cv::morphologyEx(dark_mask, dark_mask, cv::MORPH_CLOSE, k_v);
    cv::morphologyEx(dark_mask, dark_mask, cv::MORPH_CLOSE, k_h);

    int pad_short = std::min(w, h);
    int min_length = cp.min_length;
    if (min_length <= 0) min_length = std::max(pad_short / 3, 5);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(dark_mask, lines, 1, CV_PI / 180.0, cp.hough_thresh, min_length, cp.max_gap);
    if (lines.empty()) return false;

    int angle_tol = cp.angle_tol;
    std::vector<cv::Vec4i> merged;
    std::vector<bool> used(lines.size(), false);
    for (size_t i = 0; i < lines.size(); i++) {
        if (used[i]) continue;
        cv::Vec4i best = lines[i];
        used[i] = true;
        for (size_t j = i + 1; j < lines.size(); j++) {
            if (used[j]) continue;
            double dy = lines[j][1] - lines[j][3];
            double dx = lines[j][0] - lines[j][2];
            double angle_j = std::abs(std::atan2(dy, dx) * 180.0 / CV_PI);
            double dyb = best[1] - best[3];
            double dxb = best[0] - best[2];
            double angle_b = std::abs(std::atan2(dyb, dxb) * 180.0 / CV_PI);
            if (std::abs(angle_j - angle_b) > angle_tol) continue;
            double d1 = std::hypot(lines[j][0] - best[2], lines[j][1] - best[3]);
            double d2 = std::hypot(best[0] - lines[j][2], best[1] - lines[j][3]);
            if (d1 < 10 || d2 < 10) {
                best[0] = std::min(best[0], lines[j][0]);
                best[1] = std::min(best[1], lines[j][1]);
                best[2] = std::max(best[2], lines[j][2]);
                best[3] = std::max(best[3], lines[j][3]);
                used[j] = true;
            }
        }
        merged.push_back(best);
    }

    bool pad_is_vertical = h > w;
    int edge_margin = std::max(pad_short / 10, 3);
    double min_len_ratio = cp.min_len_ratio;

    for (auto& ln : merged) {
        int x1 = ln[0], y1 = ln[1], x2 = ln[2], y2 = ln[3];
        double line_angle = std::abs(std::atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI);
        if (line_angle > 90) line_angle = 180 - line_angle;

        if (pad_is_vertical && line_angle > 5) continue;
        if (!pad_is_vertical && line_angle < 85) continue;

        bool p1_edge = (x1 < edge_margin || x1 > w - edge_margin ||
                        y1 < edge_margin || y1 > h - edge_margin);
        bool p2_edge = (x2 < edge_margin || x2 > w - edge_margin ||
                        y2 < edge_margin || y2 > h - edge_margin);
        if (p1_edge && p2_edge) continue;

        double length = std::hypot(x2 - x1, y2 - y1);
        if (length > pad_short * min_len_ratio) return true;
    }
    return false;
}

// 批量crack检测: 一次调用处理多个pad ROI
py::list detect_crack_batch_fast(
    py::array_t<uint8_t> gray_full,
    py::list roi_list,
    py::dict p)
{
    auto g_buf = gray_full.request();
    cv::Mat gray_img(g_buf.shape[0], g_buf.shape[1], CV_8UC1, g_buf.ptr);

    int n = (int)roi_list.size();
    std::vector<bool> results(n, false);

    // 先在并行区域外提取ROI数据和参数 (Python API不是线程安全的)
    CrackParams cp = _extract_crack_params(p);
    std::vector<cv::Rect> rois(n);
    for (int i = 0; i < n; i++) {
        auto roi = roi_list[i].cast<py::tuple>();
        int rx = roi[0].cast<int>();
        int ry = roi[1].cast<int>();
        int rw = roi[2].cast<int>();
        int rh = roi[3].cast<int>();
        rois[i] = cv::Rect(rx, ry, rw, rh);
    }

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < n; idx++) {
        cv::Rect r = rois[idx];
        int x1c = std::max(0, r.x);
        int y1c = std::max(0, r.y);
        int x2c = std::min(gray_img.cols, r.x + r.width);
        int y2c = std::min(gray_img.rows, r.y + r.height);
        int pw = x2c - x1c;
        int ph = y2c - y1c;

        if (pw < 10 || ph < 10) continue;

        cv::Mat gray_roi = gray_img(cv::Rect(x1c, y1c, pw, ph));
        cv::Mat pad_mask_roi = cv::Mat::ones(ph, pw, CV_8UC1) * 255;

        results[idx] = _detect_crack_single(gray_roi, pad_mask_roi, cp);
    }

    py::list out;
    for (int i = 0; i < n; i++) {
        out.append(py::bool_(results[i]));
    }
    return out;
}

// ============================================================
// Pybind11 模块绑定
// ============================================================
PYBIND11_MODULE(_fast_core, m) {
    m.doc() = "C++ accelerated operations for solder detection";

    m.def("compute_diff_fast", &compute_diff_fast,
          py::arg("template_img"), py::arg("inspect_img"),
          py::arg("info"), py::arg("params"),
          "Fast compute_diff: warp + absdiff + threshold + morph + filter");

    m.def("detect_crack_hough_fast", &detect_crack_hough_fast,
          py::arg("gray_roi"), py::arg("pad_mask_roi"), py::arg("p"),
          "Fast crack detection via HoughLinesP with full filtering pipeline");

    m.def("detect_crack_batch_fast", &detect_crack_batch_fast,
          py::arg("gray_full"), py::arg("roi_list"), py::arg("p"),
          "Batch crack detection for multiple pad ROIs (OpenMP parallel)");
}
