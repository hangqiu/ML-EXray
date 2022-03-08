//
// Created by hang on 5/27/21.
//

#ifndef EDGEMLINSIGHT_EDGEML_MONITOR_NATIVE_H
#define EDGEMLINSIGHT_EDGEML_MONITOR_NATIVE_H

#include "fstream"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/model.h"
#include "edgeml/cpp/tfbenchmark/benchmark_model.h"
#include "edgeml/cpp/tfbenchmark/benchmark_tflite_model.h"
#include "edgeml/cpp/tfbenchmark/stats_calculator.h"
#include "opencv2/core.hpp"
#include "string.h"

using namespace cv;
using namespace std;

class EdgeMLMonitor_Native {
private:
    const char *TAG = "EdgeMLnative";
    long modelSize = 0;
    tflite::profiling::memory::MemoryUsage
            init_start_mem_usage, init_end_mem_usage, init_mem_usage,
            inference_start_mem_usage, inference_end_mem_usage, inference_mem_usage,
            overall_mem_usage;
    int64_t initialization_start_us, initialization_end_us, startup_latency_us;
    int64_t inference_start_us, inference_end_us, inference_latency_us;
    int64_t frame_start_us, frame_end_us, frame_latency_us;
    tflite::benchmark::BenchmarkTfLiteModel *m_Benchmark;
    std::ofstream native_log_stream;
    edgeml::Stat<int64_t> run_stats;
    string logDir;
    string inputDir;
    string logFile;
    string inputFile;
    int frameID = 0;
    int SummaryPerFrame = 5;
    double model_size_mb;
    uint64_t input_bytes;
    Mat inputImage;
    bool inputImageUpdated = false;
    std::map<std::string, uint64_t> op_start_time_us, op_end_time_us;
    std::vector<float> model_input;
    std::vector<float> model_output;
    std::vector<float> embeddings;
    std::vector<float> embeddings_dims;
    std::vector<std::string> layer_output_name;
    std::vector<std::vector<float>> layer_output;
    std::vector<std::vector<float>> layer_output_dims;

    bool playback = false;

    bool debugging_logging = false;
    bool no_logging = false;
    bool per_layer_logging = false;
    bool embedding_logging = false;
    bool input_logging = false;
    bool output_logging = false;

public:
    EdgeMLMonitor_Native(string logFile, bool playback_,
                         bool no_logging,
                         bool per_layer_logging,
                         bool embedding_logging,
                         bool input_logging,
                         bool output_logging);

    ~EdgeMLMonitor_Native();

    void initTFLiteBenchmarkModel(std::unique_ptr<tflite::Interpreter> *interpreter_);

    void endTFLiteBenchmarkModel();

    void onTFliteInitStart(long modelSize_ = 0);

    void onTFliteInitStop(std::unique_ptr<tflite::Interpreter> *interpreter_);

    void onTFliteInferenceStart();

    void onTFliteInferenceStart(Mat image);

    void onTFliteInferenceStop();

    void onTFliteInferenceStop(std::unique_ptr<tflite::Interpreter> *m_interpreter,
                               TfLiteTensor *input_tensor,
                               TfLiteTensor *embedding_layer_output = nullptr,
                               TfLiteTensor *output_tensor = nullptr, bool quantized = false);

    std::vector<std::vector<float>>
    log_tensor(TfLiteTensor *tensor_ptr, bool quantized, bool uint8_type, int tensor_id, string tensor_name);

    void log_span_to_file(int layer_id,
                          const char* tensor_name,
                          float *output_dim,
                          int output_dim_size,
                          float *output_val,
                          int output_val_size,
                          uint64_t layer_end_time);

    void onRandomOpsStart(std::string opName);

    void onRandomOpsStop(std::string opName);

    void startCurrentFrame();
    void endCurrentFrame();

};

#endif //EDGEMLINSIGHT_EDGEML_MONITOR_NATIVE_H
