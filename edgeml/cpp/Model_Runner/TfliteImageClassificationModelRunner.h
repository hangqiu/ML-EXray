//
// Created by hang on 1/13/22.
//

#ifndef EDGEMLINSIGHT_TFLITEIMAGECLASSIFICATIONMODELRUNNER_H
#define EDGEMLINSIGHT_TFLITEIMAGECLASSIFICATIONMODELRUNNER_H

#include <opencv2/imgproc.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "edgeml/cpp/tflite/edgeml_monitor_native/edgeml_monitor_native.h"

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


class TfliteImageClassificationModelRunner {
public:
    TfliteImageClassificationModelRunner(
            int input_size,
            bool quantized,
            bool per_layer_logging = true,
            std::string resizing_func_choice = "BILINEAR",
            std::string channel_choice = "RGB",
            std::string input_scale_range = "MINUSONE_ONE");

    void init(const char *tflite_model_path);

    void clear();

    std::vector<ClassificationResult> invoke_model(cv::Mat input);

private:
    const char *TAG = "ImageClassifierModelRunner";
    const int DETECTION_MODEL_CNLS = 3;
    const int DETECT_NUM = 3;
    const float IMAGE_MEAN = 128.0;
    const float IMAGE_STD = 128.0;
    const int NUM_CLASS = 1000; // was 1001 for tf hub models, 1000 for ckpt converted models

    int _input_size;
    bool _quantized;
    bool _per_layer_logging;
    std::string _resizing_func_choice;
    std::string _channel_choice;
    std::string _input_scale_range;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

    TfLiteTensor *m_input_tensor = nullptr;
    TfLiteTensor *m_output_scores = nullptr;
    TfLiteTensor *m_last_tensor = nullptr;

    EdgeMLMonitor_Native *mlMonitorNative;
};


#endif //EDGEMLINSIGHT_TFLITEIMAGECLASSIFICATIONMODELRUNNER_H
