// YOLOv8n live display + ByteTrack-style tracking for RK3588.
//
// Usage:
//   ./rknn_yolov8_bytetrack_live_demo model/yolov8n_runtime152.rknn rtsp://user:pass@ip:554/stream1

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <thread>
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
    cv::Point2f velocity;
    int cls_id;
    float score;
    int age;
    int missed;
    float total_motion;
    int static_frames;
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

class LatestFrameReader
{
public:
    explicit LatestFrameReader(cv::VideoCapture &capture) : cap_(capture), running_(false), frame_seq_(0)
    {
    }

    void start()
    {
        running_ = true;
        worker_ = std::thread(&LatestFrameReader::loop, this);
    }

    void stop()
    {
        running_ = false;
        if (worker_.joinable())
        {
            worker_.join();
        }
    }

    bool read(cv::Mat &frame, int &seq)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latest_frame_.empty())
        {
            return false;
        }
        latest_frame_.copyTo(frame);
        seq = frame_seq_;
        return true;
    }

private:
    void loop()
    {
        cv::Mat frame;
        while (running_ && g_running)
        {
            if (!cap_.read(frame) || frame.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                frame.copyTo(latest_frame_);
                frame_seq_++;
            }
        }
    }

    cv::VideoCapture &cap_;
    std::atomic<bool> running_;
    std::thread worker_;
    std::mutex mutex_;
    cv::Mat latest_frame_;
    int frame_seq_;
};

static void handle_signal(int)
{
    g_running = 0;
}

static std::string now_timestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm;
    localtime_r(&now_time, &local_tm);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local_tm);
    return std::string(buffer);
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

static void apply_match(Track &track, const Detection &det)
{
    cv::Point old_center = center_point(track.box);
    cv::Point new_center = center_point(det.box);
    float motion = (float)cv::norm(new_center - old_center);
    track.velocity = 0.75f * track.velocity + 0.25f * cv::Point2f((float)(new_center.x - old_center.x),
                                                                  (float)(new_center.y - old_center.y));
    track.box = det.box;
    track.cls_id = det.cls_id;
    track.score = det.score;
    track.age++;
    track.missed = 0;
    track.total_motion += motion;
    if (motion < 2.0f)
    {
        track.static_frames++;
    }
    else
    {
        track.static_frames = std::max(0, track.static_frames - 3);
    }
    track.trail.push_back(foot_point(track.box));
    if (track.trail.size() > 100)
    {
        track.trail.erase(track.trail.begin());
    }
}

static void match_detections(std::vector<Track> &tracks,
                             const std::vector<int> &track_indices,
                             const std::vector<Detection> &detections,
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
                                    const std::vector<Detection> &high_detections,
                                    const std::vector<Detection> &low_detections)
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

    // ByteTrack pass 1: match existing tracks with confident detections.
    match_detections(tracks, all_track_indices, high_detections, 0.30f, track_assigned, high_assigned);

    // ByteTrack pass 2: recover unmatched tracks using low-score detections only.
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

    // New IDs are created only from high-score detections.
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
        track.total_motion = 0.0f;
        track.static_frames = 0;
        track.trail.push_back(foot_point(track.box));
        tracks.push_back(track);
    }

    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [max_missed](const Track &track) {
        return track.missed > max_missed;
    }), tracks.end());
}

static bool is_static_suppressed(const Track &track)
{
    return track.age > 75 &&
           track.total_motion < 35.0f &&
           track.static_frames > 55;
}

static void draw_tracks(cv::Mat &frame, const std::vector<Track> &tracks)
{
    for (const Track &track : tracks)
    {
        if (track.missed > 0)
        {
            continue;
        }

        if (is_static_suppressed(track))
        {
            continue;
        }

        cv::Scalar color = track_color(track.id);
        cv::rectangle(frame, track.box, color, 2);

        char label[112];
        snprintf(label, sizeof(label), "ID %d ByteTrack %s %.0f%%",
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

static int count_visible_tracks(const std::vector<Track> &tracks)
{
    int visible = 0;
    for (const Track &track : tracks)
    {
        if (track.missed > 0)
        {
            continue;
        }
        if (!is_static_suppressed(track))
        {
            visible++;
        }
    }
    return visible;
}

static void write_tracks_csv(std::ofstream &csv, const std::string &timestamp,
                             int frame_index, const std::vector<Track> &tracks)
{
    for (const Track &track : tracks)
    {
        bool visible = track.missed == 0 && !is_static_suppressed(track);
        cv::Point foot = foot_point(track.box);
        cv::Point center = center_point(track.box);

        csv << timestamp << ','
            << frame_index << ','
            << track.id << ','
            << track.cls_id << ','
            << coco_cls_to_name(track.cls_id) << ','
            << track.score << ','
            << (int)track.box.x << ','
            << (int)track.box.y << ','
            << (int)(track.box.x + track.box.width) << ','
            << (int)(track.box.y + track.box.height) << ','
            << center.x << ','
            << center.y << ','
            << foot.x << ','
            << foot.y << ','
            << track.age << ','
            << track.missed << ','
            << track.total_motion << ','
            << track.static_frames << ','
            << (visible ? 1 : 0)
            << '\n';
    }
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4)
    {
        printf("%s <model_path> <camera|rtsp|input.mp4> [track_log.csv]\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *input_arg = argv[2];
    const char *csv_path = argc == 4 ? argv[3] : "/tmp/yolov8n_bytetrack_tracks.csv";

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
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }

    if (!cap.isOpened())
    {
        printf("open input fail: %s\n", input_arg);
        release_yolov8_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    std::ofstream track_csv(csv_path);
    if (!track_csv.is_open())
    {
        printf("open track csv fail: %s\n", csv_path);
        release_yolov8_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }
    track_csv << "timestamp,frame,track_id,class_id,class_name,confidence,"
              << "x1,y1,x2,y2,cx,cy,foot_x,foot_y,age,missed,total_motion,static_frames,visible\n";

    const char *window_name = "RK3588 YOLOv8n ByteTrack";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    cv::Mat bgr_frame;
    cv::Mat rgb_frame;
    object_detect_result_list od_results;
    std::vector<Track> tracks;
    int frame_index = 0;
    int last_frame_seq = -1;
    int dropped_reader_frames = 0;
    auto fps_start = std::chrono::steady_clock::now();

    printf("YOLOv8n ByteTrack-style tracking started on %s\n", input_arg);
    printf("track csv: %s\n", csv_path);
    printf("RTSP low-latency mode: CAP_PROP_BUFFERSIZE=1, latest-frame reader thread enabled\n");

    LatestFrameReader reader(cap);
    reader.start();

    while (g_running)
    {
        int frame_seq = 0;
        if (!reader.read(bgr_frame, frame_seq) || bgr_frame.empty())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (frame_seq == last_frame_seq)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (last_frame_seq >= 0 && frame_seq > last_frame_seq + 1)
        {
            dropped_reader_frames += frame_seq - last_frame_seq - 1;
        }
        last_frame_seq = frame_seq;

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

        std::vector<Detection> high_detections;
        std::vector<Detection> low_detections;
        for (int i = 0; i < od_results.count; ++i)
        {
            object_detect_result *det = &(od_results.results[i]);
            if (det->cls_id != 0 || det->prop < 0.10f)
            {
                continue;
            }

            int x1 = std::max(0, det->box.left);
            int y1 = std::max(0, det->box.top);
            int x2 = std::min(rgb_frame.cols - 1, det->box.right);
            int y2 = std::min(rgb_frame.rows - 1, det->box.bottom);
            if (x2 <= x1 || y2 <= y1)
            {
                continue;
            }

            Detection detection;
            detection.box = cv::Rect2f((float)x1, (float)y1,
                                       (float)(x2 - x1), (float)(y2 - y1));
            detection.cls_id = det->cls_id;
            detection.score = det->prop;

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
        write_tracks_csv(track_csv, now_timestamp(), frame_index, tracks);
        if (frame_index % 30 == 0)
        {
            track_csv.flush();
        }
        draw_tracks(bgr_frame, tracks);

        if (frame_index > 0 && frame_index % 30 == 0)
        {
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - fps_start).count();
            int visible_tracks = count_visible_tracks(tracks);
            printf("bytetrack_perf frame=%d fps=%.2f high=%zu low=%zu tracks=%zu visible=%d dropped=%d reader_seq=%d\n",
                   frame_index, 30.0 / elapsed_sec, high_detections.size(),
                   low_detections.size(), tracks.size(), visible_tracks, dropped_reader_frames, frame_seq);
            fflush(stdout);
            dropped_reader_frames = 0;
            fps_start = now;
        }

        char status[160];
        int visible_tracks = count_visible_tracks(tracks);
        snprintf(status, sizeof(status), "YOLOv8n + ByteTrack  visible IDs: %d / %zu  high: %zu  low: %zu",
                 visible_tracks, tracks.size(), high_detections.size(), low_detections.size());
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

    reader.stop();
    track_csv.close();
    cv::destroyAllWindows();
    release_yolov8_model(&rknn_app_ctx);
    deinit_post_process();

    return 0;
}
