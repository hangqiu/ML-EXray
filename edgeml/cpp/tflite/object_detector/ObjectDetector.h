#pragma once

#include <opencv2/core.hpp>
//#include "mediapipe/framework/port/opencv_core_inc.h"
#include "tensorflow/lite/model.h"
#include "edgeml/cpp/tflite/edgeml_monitor_native/edgeml_monitor_native.h"

using namespace cv;

struct DetectResult {
    int label = -1;
    float score = 0;
    float ymin = 0.0;
    float xmin = 0.0;
    float ymax = 0.0;
    float xmax = 0.0;
};

class ObjectDetector {
public:
    ObjectDetector(const char *tfliteModel,
                   string nativeLogDir,
                   long modelSize,
                   bool playback_,
                   bool quantized,
                   int model_input_size,
                   string acceleartor_choice,
                   int numThreads,
                   string logging_choice,
                   string resizing_func_choice,
                   string scale_range_choice,
                   string channel_choice
                   );

    ~ObjectDetector();

    DetectResult *detect(Mat src);

    const int DETECT_NUM = 3;
private:
    // members
    const char *TAG = "ObjectDetector";
    const int DETECTION_MODEL_CNLS = 3;
    const float IMAGE_MEAN = 128.0;
    const float IMAGE_STD = 128.0;
    int DETECTION_MODEL_SIZE = 300;
    bool m_modelQuantized = false;
    bool m_hasDetectionModel = false;

    bool use_GPUDelegate = false;
    string m_resizing_func_choice="";
    string m_channel_choice="";
    string m_input_scale_range="";

    bool no_logging_ = false;
    bool per_layer_logging_ = false;
    bool embedding_logging_ = false;
    bool input_logging_ = false;
    bool output_logging_ = false;

    char *m_modelBytes = nullptr;
    std::unique_ptr<tflite::FlatBufferModel> m_model;
    std::unique_ptr<tflite::Interpreter> m_interpreter;
    TfLiteTensor *m_input_tensor = nullptr;
    TfLiteTensor *m_output_tensor = nullptr;
    TfLiteTensor *m_output_locations = nullptr;
    TfLiteTensor *m_output_classes = nullptr;
    TfLiteTensor *m_output_scores = nullptr;
    TfLiteTensor *m_num_detections = nullptr;
    // embeddings
    TfLiteTensor *m_last_tensor = nullptr;


    int m_numThreads = 1;
    EdgeMLMonitor_Native *mlMonitorNative;

    // Methods
    void initDetectionModel(const char *tfliteModel, long modelSize);
};