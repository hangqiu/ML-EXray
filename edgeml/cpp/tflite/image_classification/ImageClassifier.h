#pragma once

#include <opencv2/core.hpp>
//#include "mediapipe/framework/port/opencv_core_inc.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/c/common.h"
#include "edgeml/cpp/tflite/edgeml_monitor_native/edgeml_monitor_native.h"

using namespace cv;

struct ClassificationResult {
    int label = -1;
    float score = 0;

    bool operator<(const struct ClassificationResult &cr) const {
        return score < cr.score;
    }

    bool operator>(const struct ClassificationResult &cr) const {
        return score > cr.score;
    }
};

class ImageClassifier {
public:
    ImageClassifier(const char *tfliteModel, string nativeLogDir, long modelSize,
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
    ~ImageClassifier();

    ClassificationResult *classify(Mat src);

    const int DETECT_NUM = 3;
private:
    // members
    const char *TAG = "ImageClassifier";
    const int DETECTION_MODEL_CNLS = 3;
    const float IMAGE_MEAN = 128.0;
    const float IMAGE_STD = 128.0;
    const int NUM_CLASS = 1000; // was 1001 for tf hub models, 1000 for ckpt converted models
    int DETECTION_MODEL_SIZE = 224;
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
    TfLiteDelegate *m_delegate; // Do not use a unique pointer, as it doesn't know how to delete struct
    TfLiteTensor *m_input_tensor = nullptr;
    TfLiteTensor *m_output_scores = nullptr;
    // embeddings
    TfLiteTensor *m_last_tensor = nullptr;


    int m_numThreads = 1;
    EdgeMLMonitor_Native *mlMonitorNative;

    // Methods
    void initClassificationModel(const char *tfliteModel, long modelSize);
};