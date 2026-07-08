// YOLOv8n detection + lightweight SORT-style tracking + CSV export.
//
// Usage:
//   ./rknn_yolov8_tracker_csv_demo model/yolov8n_runtime152.rknn input.mp4 tracks.csv output.mp4
//   ./rknn_yolov8_tracker_csv_demo model/yolov8n_runtime152.rknn /dev/video0 tracks.csv

#include <algorithm>
#include <fstream>
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
};

struct Track
{
    int id;
    cv::Rect2f box;
    int cls_id;
    float score;
    int age;
    int missed;
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

static float iou(const cv::Rect2f &a, const cv::Rect2f &b)
{
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

static cv::Scalar track_color(int id)
{
    static const cv::Scalar colors[] = {
        cv::Scalar(255, 120, 40), cv::Scalar(80, 220, 120),
        cv::Scalar(80, 180, 255), cv::Scalar(230, 120, 255),
        cv::Scalar(255, 220, 60), cv::Scalar(120, 255, 255)};
    return colors[id % (sizeof(colors) / sizeof(colors[0]))];
}

static cv::Point foot_point(const cv::Rect2f &box)
{
    return cv::Point((int)(box.x + box.width * 0.5f), (int)(box.y + box.height));
}

static void update_tracks(std::vector<Track> &tracks, const std::vector<Detection> &detections)
{
    const float iou_threshold = 0.30f;
    const int max_missed = 15;

    std::vector<int> det_assigned(detections.size(), 0);
    std::vector<int> track_assigned(tracks.size(), 0);

    struct Match
    {
        int track;
        int det;
        float score;
    };

    std::vector<Match> candidates;
    for (size_t t = 0; t < tracks.size(); ++t)
    {
        for (size_t d = 0; d < detections.size(); ++d)
        {
            if (tracks[t].cls_id != detections[d].cls_id)
            {
                continue;
            }

            float score = iou(tracks[t].box, detections[d].box);
            if (score >= iou_threshold)
            {
                candidates.push_back({(int)t, (int)d, score});
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
        track.box = detections[m.det].box;
        track.cls_id = detections[m.det].cls_id;
        track.score = detections[m.det].score;
        track.age++;
        track.missed = 0;
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
        track.cls_id = detections[d].cls_id;
        track.score = detections[d].score;
        track.age = 1;
        track.missed = 0;
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

        char label[96];
        snprintf(label, sizeof(label), "ID %d %s %.0f%%",
                 track.id, coco_cls_to_name(track.cls_id), track.score * 100.0f);
        int y = std::max(20, (int)track.box.y - 8);
        cv::putText(frame, label, cv::Point((int)track.box.x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

        for (size_t i = 1; i < track.trail.size(); ++i)
        {
            cv::line(frame, track.trail[i - 1], track.trail[i], color, 2);
        }

        if (!track.trail.empty())
        {
            cv::circle(frame, track.trail.back(), 4, color, -1);
        }
    }
}

static void write_tracks_csv(std::ofstream &csv, int frame_index, double time_sec,
                             const std::vector<Track> &tracks)
{
    for (const Track &track : tracks)
    {
        if (track.missed > 0)
        {
            continue;
        }

        cv::Point foot = foot_point(track.box);
        csv << frame_index << ','
            << time_sec << ','
            << track.id << ','
            << track.cls_id << ','
            << coco_cls_to_name(track.cls_id) << ','
            << track.score << ','
            << (int)track.box.x << ','
            << (int)track.box.y << ','
            << (int)(track.box.x + track.box.width) << ','
            << (int)(track.box.y + track.box.height) << ','
            << foot.x << ','
            << foot.y << '\n';
    }
}

int main(int argc, char **argv)
{
    if (argc < 4 || argc > 5)
    {
        printf("%s <model_path> <camera|input.mp4> <tracks.csv> [output.mp4]\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *input_arg = argv[2];
    const char *csv_path = argv[3];
    const char *output_path = argc == 5 ? argv[4] : NULL;

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
        cap.open(input_arg);
    }

    if (!cap.isOpened())
    {
        printf("open input fail: %s\n", input_arg);
        release_yolov8_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    std::ofstream csv(csv_path);
    if (!csv.is_open())
    {
        printf("open csv fail: %s\n", csv_path);
        release_yolov8_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }
    csv << "frame,time_sec,track_id,class_id,class_name,score,x1,y1,x2,y2,foot_x,foot_y\n";

    cv::VideoWriter writer;
    cv::Mat bgr_frame;
    cv::Mat rgb_frame;
    object_detect_result_list od_results;
    std::vector<Track> tracks;

    int frame_index = 0;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1.0 || fps > 120.0)
    {
        fps = 25.0;
    }

    printf("YOLOv8 detection tracker CSV demo started on %s\n", input_arg);

    while (g_running)
    {
        if (!cap.read(bgr_frame) || bgr_frame.empty())
        {
            printf("input read finished or failed\n");
            break;
        }

        if (output_path && !writer.isOpened())
        {
            writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        fps, bgr_frame.size());
            if (!writer.isOpened())
            {
                printf("open output fail: %s\n", output_path);
                output_path = NULL;
            }
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
                detections.push_back(detection);
            }
        }

        update_tracks(tracks, detections);

        double time_sec = frame_index / fps;
        write_tracks_csv(csv, frame_index, time_sec, tracks);
        draw_tracks(bgr_frame, tracks);

        char status[128];
        snprintf(status, sizeof(status), "YOLOv8n det  tracks: %zu  detections: %zu",
                 tracks.size(), detections.size());
        cv::putText(bgr_frame, status, cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(40, 255, 40), 2);

        if (writer.isOpened())
        {
            writer.write(bgr_frame);
        }

        frame_index++;
    }

    if (writer.isOpened())
    {
        writer.release();
    }
    csv.close();

    printf("processed frames: %d\n", frame_index);
    release_yolov8_model(&rknn_app_ctx);
    deinit_post_process();

    return 0;
}
