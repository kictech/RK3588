// YOLOv8n live display + DeepSORT-style ReID tracking for RK3588.
//
// This version uses YOLOv8n for person detection and a lightweight
// appearance embedding from each person crop for ReID-style matching.
//
// Usage:
//   ./rknn_yolov8_deepsort_live_demo model/yolov8n_runtime152.rknn rtsp://user:pass@ip:554/stream1

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolov8.h"

struct Detection
{
    cv::Rect2f box;
    int cls_id;
    float score;
    std::vector<float> embedding;
};

struct Track
{
    int id;
    cv::Rect2f box;
    cv::Point2f velocity;
    int cls_id;
    float score;
    int age;
    int missed;
    std::vector<float> embedding;
    std::vector<cv::Point> trail;
};

static volatile sig_atomic_t g_running = 1;
static int g_next_track_id = 1;

static void handle_signal(int)
{
    g_running = 0;
}

static bool parse_camera_index(const char *arg, int *index)
{
    if (strncmp(arg, "/dev/video", 10) == 0)
    {
        *index = atoi(arg + 10);
        return true;
    }

    char *end = NULL;
    long value = strtol(arg, &end, 10);
    if (end != arg && *end == '\0' && value >= 0)
    {
        *index = (int)value;
        return true;
    }

    return false;
}

static cv::Rect clip_rect(const cv::Rect2f &box, const cv::Size &size)
{
    int x1 = std::max(0, (int)std::floor(box.x));
    int y1 = std::max(0, (int)std::floor(box.y));
    int x2 = std::min(size.width - 1, (int)std::ceil(box.x + box.width));
    int y2 = std::min(size.height - 1, (int)std::ceil(box.y + box.height));
    return cv::Rect(cv::Point(x1, y1), cv::Point(std::max(x1 + 1, x2), std::max(y1 + 1, y2)));
}

static float iou(const cv::Rect2f &a, const cv::Rect2f &b)
{
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

static cv::Point center_point(const cv::Rect2f &box)
{
    return cv::Point((int)(box.x + box.width * 0.5f), (int)(box.y + box.height * 0.5f));
}

static cv::Point foot_point(const cv::Rect2f &box)
{
    return cv::Point((int)(box.x + box.width * 0.5f), (int)(box.y + box.height));
}

static float cosine_similarity(const std::vector<float> &a, const std::vector<float> &b)
{
    if (a.empty() || a.size() != b.size())
    {
        return 0.0f;
    }

    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    if (na <= 1e-6f || nb <= 1e-6f)
    {
        return 0.0f;
    }

    return dot / std::sqrt(na * nb);
}

static std::vector<float> extract_reid_embedding(const cv::Mat &frame, const cv::Rect2f &box)
{
    cv::Rect roi = clip_rect(box, frame.size());
    cv::Mat crop = frame(roi).clone();
    cv::resize(crop, crop, cv::Size(64, 128));
    cv::Mat hsv;
    cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

    std::vector<float> feature;
    const int stripes = 4;
    const int h_bins = 16;
    const int s_bins = 8;
    feature.reserve(stripes * (h_bins + s_bins));

    for (int stripe = 0; stripe < stripes; ++stripe)
    {
        int y0 = stripe * hsv.rows / stripes;
        int y1 = (stripe + 1) * hsv.rows / stripes;
        std::vector<float> h_hist(h_bins, 0.0f);
        std::vector<float> s_hist(s_bins, 0.0f);

        for (int y = y0; y < y1; ++y)
        {
            const cv::Vec3b *row = hsv.ptr<cv::Vec3b>(y);
            for (int x = 0; x < hsv.cols; ++x)
            {
                int h = std::min(h_bins - 1, row[x][0] * h_bins / 180);
                int s = std::min(s_bins - 1, row[x][1] * s_bins / 256);
                h_hist[h] += 1.0f;
                s_hist[s] += 1.0f;
            }
        }

        float h_sum = std::accumulate(h_hist.begin(), h_hist.end(), 0.0f) + 1e-6f;
        float s_sum = std::accumulate(s_hist.begin(), s_hist.end(), 0.0f) + 1e-6f;
        for (float v : h_hist)
        {
            feature.push_back(v / h_sum);
        }
        for (float v : s_hist)
        {
            feature.push_back(v / s_sum);
        }
    }

    return feature;
}

static cv::Rect2f predict_box(const Track &track)
{
    cv::Rect2f predicted = track.box;
    predicted.x += track.velocity.x;
    predicted.y += track.velocity.y;
    return predicted;
}

static cv::Scalar track_color(int id)
{
    static const cv::Scalar colors[] = {
        cv::Scalar(255, 120, 40), cv::Scalar(80, 220, 120),
        cv::Scalar(80, 180, 255), cv::Scalar(230, 120, 255),
        cv::Scalar(255, 220, 60), cv::Scalar(120, 255, 255),
        cv::Scalar(255, 90, 120), cv::Scalar(180, 255, 90)};
    return colors[id % (sizeof(colors) / sizeof(colors[0]))];
}

static void update_tracks_deepsort(std::vector<Track> &tracks, const std::vector<Detection> &detections)
{
    const float min_match_score = 0.38f;
    const int max_missed = 25;

    std::vector<int> det_assigned(detections.size(), 0);
    std::vector<int> track_assigned(tracks.size(), 0);

    struct Match
    {
        int track;
        int det;
        float score;
        float iou_score;
        float appearance_score;
    };

    std::vector<Match> candidates;
    for (size_t t = 0; t < tracks.size(); ++t)
    {
        cv::Rect2f predicted = predict_box(tracks[t]);
        cv::Point tc = center_point(predicted);
        float diag = std::sqrt(predicted.width * predicted.width + predicted.height * predicted.height) + 1.0f;

        for (size_t d = 0; d < detections.size(); ++d)
        {
            if (tracks[t].cls_id != detections[d].cls_id)
            {
                continue;
            }

            float iou_score = iou(predicted, detections[d].box);
            float appearance_score = cosine_similarity(tracks[t].embedding, detections[d].embedding);
            cv::Point dc = center_point(detections[d].box);
            float center_dist = (float)cv::norm(tc - dc);
            float center_score = std::max(0.0f, 1.0f - center_dist / (diag * 2.0f));

            float score = 0.45f * iou_score + 0.45f * appearance_score + 0.10f * center_score;
            if (score >= min_match_score || (appearance_score > 0.82f && center_score > 0.35f))
            {
                candidates.push_back({(int)t, (int)d, score, iou_score, appearance_score});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Match &a, const Match &b) {
        return a.score > b.score;
    });

    for (const Match &m : candidates)
    {
        if (track_assigned[m.track] || det_assigned[m.det])
        {
            continue;
        }

        Track &track = tracks[m.track];
        cv::Point old_center = center_point(track.box);
        cv::Point new_center = center_point(detections[m.det].box);
        track.velocity = 0.7f * track.velocity + 0.3f * cv::Point2f((float)(new_center.x - old_center.x),
                                                                     (float)(new_center.y - old_center.y));
        track.box = detections[m.det].box;
        track.cls_id = detections[m.det].cls_id;
        track.score = detections[m.det].score;
        track.age++;
        track.missed = 0;

        if (track.embedding.size() == detections[m.det].embedding.size())
        {
            for (size_t i = 0; i < track.embedding.size(); ++i)
            {
                track.embedding[i] = 0.85f * track.embedding[i] + 0.15f * detections[m.det].embedding[i];
            }
        }
        else
        {
            track.embedding = detections[m.det].embedding;
        }

        track.trail.push_back(foot_point(track.box));
        if (track.trail.size() > 100)
        {
            track.trail.erase(track.trail.begin());
        }

        track_assigned[m.track] = 1;
        det_assigned[m.det] = 1;
    }

    for (size_t t = 0; t < tracks.size(); ++t)
    {
        if (!track_assigned[t])
        {
            tracks[t].missed++;
            tracks[t].box = predict_box(tracks[t]);
        }
    }

    for (size_t d = 0; d < detections.size(); ++d)
    {
        if (det_assigned[d])
        {
            continue;
        }

        Track track;
        track.id = g_next_track_id++;
        track.box = detections[d].box;
        track.velocity = cv::Point2f(0.0f, 0.0f);
        track.cls_id = detections[d].cls_id;
        track.score = detections[d].score;
        track.age = 1;
        track.missed = 0;
        track.embedding = detections[d].embedding;
        track.trail.push_back(foot_point(track.box));
        tracks.push_back(track);
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [max_missed](const Track &track) {
        return track.missed > max_missed;
    }), tracks.end());
}

static void draw_tracks(cv::Mat &frame, const std::vector<Track> &tracks)
{
    for (const Track &track : tracks)
    {
        if (track.missed > 0)
        {
            continue;
        }

        cv::Scalar color = track_color(track.id);
        cv::rectangle(frame, track.box, color, 2);

        char label[112];
        snprintf(label, sizeof(label), "ID %d ReID %s %.0f%%",
                 track.id, coco_cls_to_name(track.cls_id), track.score * 100.0f);

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        int x = std::max(0, (int)track.box.x);
        int y = std::max(label_size.height + 8, (int)track.box.y);
        cv::rectangle(frame,
                      cv::Point(x, y - label_size.height - 8),
                      cv::Point(std::min(frame.cols - 1, x + label_size.width + 8), y + baseline),
                      color, -1);
        cv::putText(frame, label, cv::Point(x + 4, y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

        for (size_t i = 1; i < track.trail.size(); ++i)
        {
            cv::line(frame, track.trail[i - 1], track.trail[i], color, 2);
        }
        cv::circle(frame, foot_point(track.box), 4, color, -1);
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <camera|rtsp|input.mp4>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *input_arg = argv[2];

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    int ret = init_yolov8_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_model fail! ret=%d model_path=%s\n", ret, model_path);
        deinit_post_process();
        return ret;
    }

    int camera_index = 0;
    bool input_is_camera = parse_camera_index(input_arg, &camera_index);

    cv::VideoCapture cap;
    if (input_is_camera)
    {
        cap.open(camera_index, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
    }
    else
    {
        cap.open(input_arg, cv::CAP_FFMPEG);
    }

    if (!cap.isOpened())
    {
        printf("open input fail: %s\n", input_arg);
        release_yolov8_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    const char *window_name = "RK3588 YOLOv8n DeepSORT ReID";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    cv::Mat bgr_frame;
    cv::Mat rgb_frame;
    object_detect_result_list od_results;
    std::vector<Track> tracks;
    int frame_index = 0;
    auto fps_start = std::chrono::steady_clock::now();

    printf("YOLOv8n DeepSORT-style ReID tracking started on %s\n", input_arg);

    while (g_running)
    {
        if (!cap.read(bgr_frame) || bgr_frame.empty())
        {
            printf("input read failed\n");
            break;
        }

        cv::cvtColor(bgr_frame, rgb_frame, cv::COLOR_BGR2RGB);

        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));
        src_image.width = rgb_frame.cols;
        src_image.height = rgb_frame.rows;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.size = rgb_frame.total() * rgb_frame.elemSize();
        src_image.virt_addr = rgb_frame.data;

        ret = inference_yolov8_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("inference_yolov8_model fail! ret=%d\n", ret);
            continue;
        }

        std::vector<Detection> detections;
        for (int i = 0; i < od_results.count; ++i)
        {
            object_detect_result *det = &(od_results.results[i]);
            if (det->cls_id != 0)
            {
                continue;
            }

            int x1 = std::max(0, det->box.left);
            int y1 = std::max(0, det->box.top);
            int x2 = std::min(rgb_frame.cols - 1, det->box.right);
            int y2 = std::min(rgb_frame.rows - 1, det->box.bottom);
            if (x2 > x1 && y2 > y1)
            {
                Detection detection;
                detection.box = cv::Rect2f((float)x1, (float)y1,
                                           (float)(x2 - x1), (float)(y2 - y1));
                detection.cls_id = det->cls_id;
                detection.score = det->prop;
                detection.embedding = extract_reid_embedding(bgr_frame, detection.box);
                detections.push_back(detection);
            }
        }

        update_tracks_deepsort(tracks, detections);
        draw_tracks(bgr_frame, tracks);

        if (frame_index > 0 && frame_index % 30 == 0)
        {
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - fps_start).count();
            printf("deepsort_perf frame=%d fps=%.2f detections=%zu tracks=%zu\n",
                   frame_index, 30.0 / elapsed_sec, detections.size(), tracks.size());
            fflush(stdout);
            fps_start = now;
        }

        char status[160];
        snprintf(status, sizeof(status), "YOLOv8n + DeepSORT-style ReID  active IDs: %zu  detections: %zu",
                 tracks.size(), detections.size());
        cv::putText(bgr_frame, status, cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(40, 255, 40), 2);

        cv::imshow(window_name, bgr_frame);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27)
        {
            break;
        }

        frame_index++;
    }

    cv::destroyAllWindows();
    release_yolov8_model(&rknn_app_ctx);
    deinit_post_process();

    return 0;
}
