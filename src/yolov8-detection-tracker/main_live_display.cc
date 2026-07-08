// YOLOv8n live display demo for RK3588 / Orange Pi 5 Plus.
//
// Usage:
//   ./rknn_yolov8_live_demo model/yolov8n_runtime152.rknn rtsp://user:pass@ip:554/stream1
//   ./rknn_yolov8_live_demo model/yolov8n_runtime152.rknn /dev/video0

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>

#include <opencv2/opencv.hpp>

#include "yolov8.h"

static volatile sig_atomic_t g_running = 1;

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

static cv::Scalar class_color(int cls_id)
{
    static const cv::Scalar colors[] = {
        cv::Scalar(40, 220, 40), cv::Scalar(80, 180, 255),
        cv::Scalar(255, 160, 40), cv::Scalar(230, 120, 255),
        cv::Scalar(255, 220, 60), cv::Scalar(120, 255, 255)};
    return colors[cls_id % (sizeof(colors) / sizeof(colors[0]))];
}

static void draw_detections(cv::Mat &frame, const object_detect_result_list &results)
{
    for (int i = 0; i < results.count; ++i)
    {
        const object_detect_result *det = &(results.results[i]);

        int x1 = std::max(0, det->box.left);
        int y1 = std::max(0, det->box.top);
        int x2 = std::min(frame.cols - 1, det->box.right);
        int y2 = std::min(frame.rows - 1, det->box.bottom);
        if (x2 <= x1 || y2 <= y1)
        {
            continue;
        }

        cv::Scalar color = class_color(det->cls_id);
        cv::rectangle(frame, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), color, 2);

        char label[128];
        snprintf(label, sizeof(label), "%s %.0f%%", coco_cls_to_name(det->cls_id), det->prop * 100.0f);

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
        int text_y = std::max(label_size.height + 6, y1);
        cv::rectangle(frame,
                      cv::Point(x1, text_y - label_size.height - 6),
                      cv::Point(std::min(frame.cols - 1, x1 + label_size.width + 6), text_y + baseline),
                      color, -1);
        cv::putText(frame, label, cv::Point(x1 + 3, text_y - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2);
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

    const char *window_name = "RK3588 YOLOv8n Detection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    cv::Mat bgr_frame;
    cv::Mat rgb_frame;
    object_detect_result_list od_results;
    int frame_index = 0;
    double read_ms_sum = 0.0;
    double cvt_ms_sum = 0.0;
    double infer_ms_sum = 0.0;
    double draw_ms_sum = 0.0;
    double show_ms_sum = 0.0;
    auto fps_start = std::chrono::steady_clock::now();

    printf("YOLOv8n live detection started on %s\n", input_arg);

    while (g_running)
    {
        auto t0 = std::chrono::steady_clock::now();
        if (!cap.read(bgr_frame) || bgr_frame.empty())
        {
            printf("input read failed\n");
            break;
        }
        auto t1 = std::chrono::steady_clock::now();

        cv::cvtColor(bgr_frame, rgb_frame, cv::COLOR_BGR2RGB);
        auto t2 = std::chrono::steady_clock::now();

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
        auto t3 = std::chrono::steady_clock::now();

        draw_detections(bgr_frame, od_results);

        char status[128];
        snprintf(status, sizeof(status), "YOLOv8n detections: %d  frame: %d", od_results.count, frame_index);
        cv::putText(bgr_frame, status, cv::Point(20, 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(40, 255, 40), 2);
        auto t4 = std::chrono::steady_clock::now();

        cv::imshow(window_name, bgr_frame);
        int key = cv::waitKey(1);
        auto t5 = std::chrono::steady_clock::now();

        read_ms_sum += std::chrono::duration<double, std::milli>(t1 - t0).count();
        cvt_ms_sum += std::chrono::duration<double, std::milli>(t2 - t1).count();
        infer_ms_sum += std::chrono::duration<double, std::milli>(t3 - t2).count();
        draw_ms_sum += std::chrono::duration<double, std::milli>(t4 - t3).count();
        show_ms_sum += std::chrono::duration<double, std::milli>(t5 - t4).count();

        if (frame_index > 0 && frame_index % 30 == 0)
        {
            auto now = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - fps_start).count();
            double n = 30.0;
            printf("perf frame=%d fps=%.2f read=%.1fms cvt=%.1fms infer=%.1fms draw=%.1fms show=%.1fms pipeline=%.1fms\n",
                   frame_index,
                   n / elapsed_sec,
                   read_ms_sum / n,
                   cvt_ms_sum / n,
                   infer_ms_sum / n,
                   draw_ms_sum / n,
                   show_ms_sum / n,
                   (read_ms_sum + cvt_ms_sum + infer_ms_sum + draw_ms_sum + show_ms_sum) / n);
            fflush(stdout);
            read_ms_sum = 0.0;
            cvt_ms_sum = 0.0;
            infer_ms_sum = 0.0;
            draw_ms_sum = 0.0;
            show_ms_sum = 0.0;
            fps_start = now;
        }

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
