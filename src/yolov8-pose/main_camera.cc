// Camera display demo for RKNN YOLOv8n-pose + ByteTrack-style ID tracking on RK3588.
//
// Usage:
//   ./rknn_yolov8_pose_demo model/yolov8n-pose.rknn /dev/video0
//   ./rknn_yolov8_pose_demo model/yolov8n-pose.rknn rtsp://user:pass@ip:554/stream1

#include <algorithm>
#include <array>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolov8-pose.h"

struct PoseDetection
{
    cv::Rect2f box;
    int cls_id;
    float score;
    std::array<cv::Point2f, 17> keypoints;
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
    std::array<cv::Point2f, 17> keypoints;
    std::vector<cv::Point> trail;
};

struct Match
{
    int track;
    int det;
    float score;
};

static volatile sig_atomic_t g_running = 1;
static int g_next_track_id = 1;

static int skeleton[38] = {
    16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
    7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};

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

static void apply_match(Track &track, const PoseDetection &det)
{
    cv::Point old_center = center_point(track.box);
    cv::Point new_center = center_point(det.box);
    track.velocity = 0.75f * track.velocity + 0.25f * cv::Point2f((float)(new_center.x - old_center.x),
                                                                  (float)(new_center.y - old_center.y));
    track.box = det.box;
    track.cls_id = det.cls_id;
    track.score = det.score;
    track.keypoints = det.keypoints;
    track.age++;
    track.missed = 0;
    track.trail.push_back(foot_point(track.box));
    if (track.trail.size() > 100)
    {
        track.trail.erase(track.trail.begin());
    }
}

static void match_detections(std::vector<Track> &tracks,
                             const std::vector<int> &track_indices,
                             const std::vector<PoseDetection> &detections,
                             float iou_threshold,
                             std::vector<int> &track_assigned,
                             std::vector<int> &det_assigned)
{
    std::vector<Match> candidates;
    for (int ti : track_indices)
    {
        cv::Rect2f predicted = predict_box(tracks[ti]);
        for (size_t d = 0; d < detections.size(); ++d)
        {
            if (det_assigned[d] || tracks[ti].cls_id != detections[d].cls_id)
            {
                continue;
            }

            float score = iou(predicted, detections[d].box);
            if (score >= iou_threshold)
            {
                candidates.push_back({ti, (int)d, score});
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

        apply_match(tracks[m.track], detections[m.det]);
        track_assigned[m.track] = 1;
        det_assigned[m.det] = 1;
    }
}

static void update_tracks_bytetrack(std::vector<Track> &tracks,
                                    const std::vector<PoseDetection> &high_detections,
                                    const std::vector<PoseDetection> &low_detections)
{
    const int max_missed = 30;

    std::vector<int> track_assigned(tracks.size(), 0);
    std::vector<int> high_assigned(high_detections.size(), 0);
    std::vector<int> low_assigned(low_detections.size(), 0);
    std::vector<int> all_track_indices;
    for (size_t i = 0; i < tracks.size(); ++i)
    {
        all_track_indices.push_back((int)i);
    }

    match_detections(tracks, all_track_indices, high_detections, 0.30f, track_assigned, high_assigned);

    std::vector<int> unmatched_track_indices;
    for (size_t i = 0; i < tracks.size(); ++i)
    {
        if (!track_assigned[i])
        {
            unmatched_track_indices.push_back((int)i);
        }
    }
    match_detections(tracks, unmatched_track_indices, low_detections, 0.22f, track_assigned, low_assigned);

    for (size_t t = 0; t < tracks.size(); ++t)
    {
        if (!track_assigned[t])
        {
            tracks[t].missed++;
            tracks[t].box = predict_box(tracks[t]);
        }
    }

    for (size_t d = 0; d < high_detections.size(); ++d)
    {
        if (high_assigned[d])
        {
            continue;
        }

        Track track;
        track.id = g_next_track_id++;
        track.box = high_detections[d].box;
        track.velocity = cv::Point2f(0.0f, 0.0f);
        track.cls_id = high_detections[d].cls_id;
        track.score = high_detections[d].score;
        track.age = 1;
        track.missed = 0;
        track.keypoints = high_detections[d].keypoints;
        track.trail.push_back(foot_point(track.box));
        tracks.push_back(track);
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [max_missed](const Track &track) {
        return track.missed > max_missed;
    }), tracks.end());
}

static void draw_pose_tracks(cv::Mat &frame, const std::vector<Track> &tracks)
{
    for (const Track &track : tracks)
    {
        if (track.missed > 0)
        {
            continue;
        }

        cv::Scalar color = track_color(track.id);
        cv::rectangle(frame, track.box, color, 2);

        for (int j = 0; j < 38 / 2; ++j)
        {
            int a = skeleton[2 * j] - 1;
            int b = skeleton[2 * j + 1] - 1;
            cv::line(frame, track.keypoints[a], track.keypoints[b], color, 2);
        }

        for (int j = 0; j < 17; ++j)
        {
            cv::circle(frame, track.keypoints[j], 3, cv::Scalar(0, 255, 255), -1);
        }

        for (size_t i = 1; i < track.trail.size(); ++i)
        {
            cv::line(frame, track.trail[i - 1], track.trail[i], color, 2);
        }
        cv::circle(frame, foot_point(track.box), 4, color, -1);

        char label[112];
        snprintf(label, sizeof(label), "ID %d ByteTrack pose %.0f%%",
                 track.id, track.score * 100.0f);

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
    }
}

static int count_visible_tracks(const std::vector<Track> &tracks)
{
    int visible = 0;
    for (const Track &track : tracks)
    {
        if (track.missed == 0)
        {
            visible++;
        }
    }
    return visible;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        printf("%s <model_path> [camera_index|/dev/video0|rtsp|input.mp4]\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *camera_arg = argc == 3 ? argv[2] : "/dev/video0";

    int camera_index = 0;
    bool input_is_camera = parse_camera_index(camera_arg, &camera_index);

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    int ret = init_yolov8_pose_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_pose_model fail! ret=%d model_path=%s\n", ret, model_path);
        deinit_post_process();
        return ret;
    }

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
        cap.open(camera_arg, cv::CAP_FFMPEG);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }
    if (!cap.isOpened())
    {
        printf("open input fail: %s\n", camera_arg);
        release_yolov8_pose_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    const char *window_name = "RK3588 YOLOv8n Pose";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    cv::Mat bgr_frame;
    cv::Mat rgb_frame;
    object_detect_result_list od_results;
    std::vector<Track> tracks;
    int frame_index = 0;

    printf("Camera pose + ByteTrack demo started on %s. Press q or Ctrl+C to quit.\n", camera_arg);

    while (g_running)
    {
        if (!cap.read(bgr_frame) || bgr_frame.empty())
        {
            printf("camera read failed\n");
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

        ret = inference_yolov8_pose_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0)
        {
            printf("inference_yolov8_pose_model fail! ret=%d\n", ret);
            continue;
        }

        cv::cvtColor(rgb_frame, bgr_frame, cv::COLOR_RGB2BGR);

        std::vector<PoseDetection> high_detections;
        std::vector<PoseDetection> low_detections;
        for (int i = 0; i < od_results.count; ++i)
        {
            object_detect_result *det = &(od_results.results[i]);
            if (det->prop < 0.10f)
            {
                continue;
            }

            int x1 = std::max(0, det->box.left);
            int y1 = std::max(0, det->box.top);
            int x2 = std::min(bgr_frame.cols - 1, det->box.right);
            int y2 = std::min(bgr_frame.rows - 1, det->box.bottom);
            if (x2 <= x1 || y2 <= y1)
            {
                continue;
            }

            PoseDetection detection;
            detection.box = cv::Rect2f((float)x1, (float)y1,
                                       (float)(x2 - x1), (float)(y2 - y1));
            detection.cls_id = det->cls_id;
            detection.score = det->prop;
            for (int j = 0; j < 17; ++j)
            {
                detection.keypoints[j] = cv::Point2f(det->keypoints[j][0], det->keypoints[j][1]);
            }

            if (det->prop >= 0.45f)
            {
                high_detections.push_back(detection);
            }
            else
            {
                low_detections.push_back(detection);
            }
        }

        update_tracks_bytetrack(tracks, high_detections, low_detections);
        draw_pose_tracks(bgr_frame, tracks);

        if (frame_index % 30 == 0)
        {
            printf("pose_bytetrack frame=%d high=%zu low=%zu tracks=%zu visible=%d\n",
                   frame_index, high_detections.size(), low_detections.size(),
                   tracks.size(), count_visible_tracks(tracks));
            fflush(stdout);
        }

        char status[160];
        snprintf(status, sizeof(status), "YOLOv8n-pose + ByteTrack  visible IDs: %d / %zu",
                 count_visible_tracks(tracks), tracks.size());
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
    release_yolov8_pose_model(&rknn_app_ctx);
    deinit_post_process();

    return 0;
}
