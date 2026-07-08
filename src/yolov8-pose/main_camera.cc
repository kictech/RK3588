// Camera display demo for RKNN YOLOv8n-pose on RK3588.
//
// Usage:
//   ./rknn_yolov8_pose_demo model/yolov8n-pose.rknn /dev/video0
//   ./rknn_yolov8_pose_demo model/yolov8n-pose.rknn rtsp://user:pass@ip:554/stream1

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv2/opencv.hpp>

#include "yolov8-pose.h"
#include "image_drawing.h"

static volatile sig_atomic_t g_running = 1;

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

static void draw_pose_results(image_buffer_t *image, const object_detect_result_list *results)
{
    char text[256];

    for (int i = 0; i < results->count; i++)
    {
        const object_detect_result *det_result = &(results->results[i]);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        snprintf(text, sizeof(text), "%s %.1f%%",
                 coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        draw_text(image, text, x1, y1 > 20 ? y1 - 20 : y1 + 5, COLOR_RED, 10);

        for (int j = 0; j < 38 / 2; ++j)
        {
            int a = skeleton[2 * j] - 1;
            int b = skeleton[2 * j + 1] - 1;
            draw_line(image,
                      (int)det_result->keypoints[a][0], (int)det_result->keypoints[a][1],
                      (int)det_result->keypoints[b][0], (int)det_result->keypoints[b][1],
                      COLOR_ORANGE, 3);
        }

        for (int j = 0; j < 17; ++j)
        {
            draw_circle(image,
                        (int)det_result->keypoints[j][0],
                        (int)det_result->keypoints[j][1],
                        2, COLOR_YELLOW, 2);
        }
    }
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

    printf("Camera pose demo started on %s. Press q or Ctrl+C to quit.\n", camera_arg);

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

        draw_pose_results(&src_image, &od_results);
        cv::cvtColor(rgb_frame, bgr_frame, cv::COLOR_RGB2BGR);

        cv::imshow(window_name, bgr_frame);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27)
        {
            break;
        }
    }

    cv::destroyAllWindows();
    release_yolov8_pose_model(&rknn_app_ctx);
    deinit_post_process();

    return 0;
}
