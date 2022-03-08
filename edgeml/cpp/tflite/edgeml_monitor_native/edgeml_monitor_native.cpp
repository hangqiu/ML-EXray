//
// Created by hang on 5/27/21.
//

#include "edgeml_monitor_native.h"
#include "edgeml/cpp/tfbenchmark/profiling_listener.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/logging.h"
//#include <android/log.h>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core.hpp"
#include <vector>

using namespace tflite;
using namespace benchmark;
using namespace cv;

string cat_string(char *a, char *b) {
    char *ret = (char *) malloc(strlen(a) + strlen(b) + 1);
    strcpy(ret, a);
    strcat(ret, b);
    return ret;
}

string frameID_to_imageName(int frameID) {
    char frame_name[9];
    std::sprintf(frame_name, "%08d", frameID);
    return frame_name;
}

EdgeMLMonitor_Native::EdgeMLMonitor_Native(string logDir_,
                                           bool playback_ = false,
                                           bool no_logging_ = false,
                                           bool per_layer_logging_ = false,
                                           bool embedding_logging_ = false,
                                           bool input_logging_ = false,
                                           bool output_logging_ = false) {
    if (per_layer_logging_) {
        string summary_log_file = logDir_ + "/summary.log";
        m_Benchmark = new BenchmarkTfLiteModel(
                BenchmarkTfLiteModel::DefaultParams(summary_log_file));
    }
    logDir = logDir_ + "/nativeLog/";
    string initLog = "/init.log";
    logFile = logDir + initLog;
    native_log_stream.
            open(logFile, std::ofstream::out
    );

    inputDir = logDir_ + "/nativeInput/";

    inputFile = inputDir + frameID_to_imageName(frameID) + ".png";
    playback = playback_;
    no_logging = no_logging_;
    per_layer_logging = per_layer_logging_;
    embedding_logging = embedding_logging_;
    input_logging = input_logging_;
    output_logging = output_logging_;
}

EdgeMLMonitor_Native::~EdgeMLMonitor_Native() {
    if (native_log_stream) native_log_stream.close();
//    if (logFile)    free(logFile);

    std::stringstream stream;
    run_stats.OutputToStream(&stream);
    string summaryLog = "/summary.log";
    logFile = logDir + summaryLog;
    native_log_stream.open(logFile, std::ofstream::app);
    native_log_stream << stream.str() << std::endl;
    native_log_stream.close();

    if (m_Benchmark) delete m_Benchmark;
}


void
EdgeMLMonitor_Native::initTFLiteBenchmarkModel(std::unique_ptr<tflite::Interpreter> *interpreter_) {
//    __android_log_print(ANDROID_LOG_ERROR, TAG, "Benchmark Model Init");
//    char* args = "--enable_op_profiling=true --profiling_output_csv_file=/profile.csv";
//    int argc = 2;
//    benchmark.ParseFlags(&argc, &args);
    m_Benchmark->LogParams();
//    model_size_mb = benchmark.MayGetModelFileSize() / 1e6;
    model_size_mb = 0;
    m_Benchmark->InitBenchmarkModel(interpreter_);
//    input_bytes = benchmark.ComputeInputBytes();
    input_bytes = 0;
    m_Benchmark->listeners_.OnBenchmarkStart(m_Benchmark->params_);
//    __android_log_print(ANDROID_LOG_ERROR, TAG, "Finished Benchmark Model Init");

}

void EdgeMLMonitor_Native::endTFLiteBenchmarkModel() {
    overall_mem_usage = profiling::memory::GetMemoryUsage() - init_start_mem_usage;
    m_Benchmark->listeners_.OnBenchmarkEnd({model_size_mb, startup_latency_us, input_bytes,
                                            run_stats, run_stats, init_mem_usage,
                                            overall_mem_usage});
}

// run before loading model and creating intepreter
void EdgeMLMonitor_Native::onTFliteInitStart(long modelSize_) {
    modelSize = modelSize_;
    if (modelSize > 0) {
        TFLITE_LOG(ERROR) << "The input model file size (MB): " << modelSize;
    }
    init_start_mem_usage = profiling::memory::GetMemoryUsage();
    initialization_start_us = profiling::time::NowMicros();
    native_log_stream << "Initialization Start time: " << initialization_start_us / 1e3
                      << "ms." << std::endl;

}

void EdgeMLMonitor_Native::onTFliteInitStop(std::unique_ptr<tflite::Interpreter> *interpreter_) {

    initialization_end_us = profiling::time::NowMicros();
    init_end_mem_usage = profiling::memory::GetMemoryUsage();
    startup_latency_us = initialization_end_us - initialization_start_us;
    init_mem_usage = init_end_mem_usage - init_start_mem_usage;
    native_log_stream << "Initialization latency: " << startup_latency_us / 1e3
                      << "ms." << std::endl;
    native_log_stream << "Initialization memeory usage: " << std::endl << init_mem_usage
                      << std::endl;
    native_log_stream.close();

    logFile = logDir + frameID_to_imageName(frameID);
}


void EdgeMLMonitor_Native::onTFliteInferenceStart() {
//    __android_log_print(ANDROID_LOG_ERROR, TAG, "InferenceStart");
    inference_start_mem_usage = profiling::memory::GetMemoryUsage();
    inference_start_us = profiling::time::NowMicros();
    if (per_layer_logging) {
        m_Benchmark->listeners_.OnSingleRunStart(tflite::benchmark::REGULAR);
    }
}

void EdgeMLMonitor_Native::onTFliteInferenceStart(Mat image) {
    if (debugging_logging) {
        image.copyTo(inputImage);
        inputImageUpdated = true;
    }
    onTFliteInferenceStart();
}


void EdgeMLMonitor_Native::onTFliteInferenceStop() {
//    __android_log_print(ANDROID_LOG_ERROR, TAG, "InferenceStop");
    inference_end_us = profiling::time::NowMicros();
    inference_end_mem_usage = profiling::memory::GetMemoryUsage();
    if (per_layer_logging) {
        m_Benchmark->listeners_.OnSingleRunEnd();
        run_stats.UpdateStat(inference_latency_us);
    }
}

std::vector<std::vector<float>>
EdgeMLMonitor_Native::log_tensor(TfLiteTensor *tensor_ptr, bool quantized, bool uint8_type = true,
                                 int tensor_id = -1, string tensor_name = "") {
    TfLiteIntArray *last_tensor_dim = tensor_ptr->dims;
    std::vector<float> tensor_dims;
    std::vector<float> tensor_vals;
    tensor_dims.clear();
    int total_count = 1;
    for (int j = 0; j < last_tensor_dim->size; j++) {
        tensor_dims.push_back(last_tensor_dim->data[j]);
        total_count *= last_tensor_dim->data[j];
    }

    void *embeddings_ptr = nullptr;
    if (quantized) {
        if (uint8_type) {
            embeddings_ptr = tensor_ptr->data.uint8;
        } else {
            embeddings_ptr = tensor_ptr->data.int8;
        }
    } else {
        embeddings_ptr = tensor_ptr->data.f;
    }
    tensor_vals.clear();
    for (int i = 0; i < total_count; i++) {
        float embed = 0.0;
        if (quantized) {
            if (uint8_type) {
                embed = ((uint8_t *) embeddings_ptr)[i];
            } else {
                embed = ((int8_t *) embeddings_ptr)[i];
            }
        } else {
            embed = ((float *) embeddings_ptr)[i];
        }
        tensor_vals.push_back(embed);
    }
    std::vector<std::vector<float>> ret;
    ret.push_back(tensor_vals);
    ret.push_back(tensor_dims);

//    std::cout<< sizeof(TfLiteTensor)<<std::endl;
    return ret;
}

void EdgeMLMonitor_Native::onTFliteInferenceStop(
        std::unique_ptr<tflite::Interpreter> *m_interpreter,
        TfLiteTensor *input_tensor_ptr,
        TfLiteTensor *embedding_layer_output,
        TfLiteTensor *output_tensor_ptr,
        bool quantized) {

    onTFliteInferenceStop();

    if (no_logging) return;
    std::vector<std::vector<float>> inputs_info, output_info, embeddings_info, layer_output_info;
    if (input_logging && input_tensor_ptr != nullptr) {
        inputs_info = log_tensor(input_tensor_ptr, quantized, true, 0, "input");
        model_input = inputs_info[0];
    }

    if (output_logging && output_tensor_ptr != nullptr) {
        output_info = log_tensor(output_tensor_ptr, quantized, true, -1, "output");
        model_output = output_info[0];
    }

    if (embedding_logging && embedding_layer_output != nullptr) {
        embeddings_info = log_tensor(embedding_layer_output, quantized, true, -1, "embedding");
        embeddings = embeddings_info[0];
        embeddings_dims = embeddings_info[1];
    }

    // per layer logging
    if (per_layer_logging) {
        string embedding_layer_name = "";
        int tensor_size = (*m_interpreter)->tensors_size();
        layer_output_name.clear();
        layer_output.clear();
        for (int j = 0; j < tensor_size; j++) {
            // last tensor is null tensor
            if (!(*m_interpreter)->tensor(j)->name)continue;
            layer_output_name.push_back((*m_interpreter)->tensor(j)->name);
            layer_output_info.clear();
            layer_output_info = log_tensor((*m_interpreter)->tensor(j), quantized, false, j,
                                           (*m_interpreter)->tensor(j)->name);
            layer_output.push_back(layer_output_info[0]);
            layer_output_dims.push_back(layer_output_info[1]);
        }
    }
}

void EdgeMLMonitor_Native::onRandomOpsStart(string opName) {
    if (no_logging) return;
    uint64_t op_start_us = profiling::time::NowMicros();
    op_start_time_us[opName] = op_start_us;
}

void EdgeMLMonitor_Native::onRandomOpsStop(string opName) {
    if (no_logging) return;
    uint64_t op_end_us = profiling::time::NowMicros();
    op_end_time_us[opName] = op_end_us;
}

void EdgeMLMonitor_Native::startCurrentFrame() {
    frame_start_us = profiling::time::NowMicros();
}

void EdgeMLMonitor_Native::log_span_to_file(int layer_id,
                                            const char* tensor_name,
                                            float *output_dim,
                                            int output_dim_size,
                                            float *output_val,
                                            int output_val_size,
                                            uint64_t layer_end_time) {
    native_log_stream << "Layer id " << layer_id << ", Layer dims [";
    for (int i=0;i<output_dim_size;i++) {
        native_log_stream << " " << output_dim[i];
    }
    native_log_stream << "] Layer end time " << layer_end_time << std::endl;

    native_log_stream << tensor_name << ":";
    for (int j=0;j<output_val_size;j++) {
        native_log_stream << " " << output_val[j];
    }
    native_log_stream << std::endl;

}

void EdgeMLMonitor_Native::endCurrentFrame() {
    // logging start
    /// random op
//    __android_log_print(ANDROID_LOG_ERROR, TAG, logFile.c_str());
    native_log_stream.open(logFile, std::ofstream::app);
    for (std::map<string, uint64_t>::iterator it = op_start_time_us.begin();
         it != op_start_time_us.end(); it++) {
        string opName = it->first;
        uint64_t op_start_us = it->second;
        native_log_stream << opName << " Start time: " << op_start_us / 1e3 << "ms" << std::endl;
        if (op_end_time_us.find(opName) != op_end_time_us.end()) {
            uint64_t op_end_us = op_end_time_us[opName];
            uint64_t op_latency_us = op_end_us - op_start_us;
            native_log_stream << opName << " Latency: " << op_latency_us / 1e3 << "ms" << std::endl;
        } else {
            native_log_stream << opName << " missing end time" << std::endl;
        }
    }
    /// inference
    inference_mem_usage = inference_end_mem_usage - inference_start_mem_usage;
    inference_latency_us = inference_end_us - inference_start_us;
    native_log_stream << "Inference Start time: " << inference_start_us / 1e3 << "ms" << std::endl;
    native_log_stream << "Inference Latency: " << inference_latency_us / 1e3 << "ms" << std::endl;
    native_log_stream << "Inference memory usage: " << std::endl << inference_mem_usage
                      << std::endl;

    /// embeddings
    if (embedding_logging) {
        native_log_stream << "Embeddings Dims: [";
        if (!embeddings_dims.empty()) {
            for (auto it = embeddings_dims.begin(); it != embeddings_dims.end(); ++it) {
                native_log_stream << " " << *it;
            }
        }
        native_log_stream << " ]" << std::endl;
        native_log_stream << "Embeddings:";
        if (!embeddings.empty()) {
            for (auto it = embeddings.begin(); it != embeddings.end(); ++it) {
                native_log_stream << " " << *it;
            }
        }
        native_log_stream << std::endl;
    }

    /// layer_output
    if (per_layer_logging) {
        native_log_stream << "Layer Outputs:" << std::endl;
        for (int i = 0; i < layer_output_name.size(); i++) {
            log_span_to_file(i, layer_output_name[i].c_str(), &layer_output_dims[i][0], layer_output_dims.size(), &layer_output[i][0],layer_output.size(), 0);
        }
    }


    /// inputs
    if (input_logging) {
        native_log_stream << "Input:";
        if (!model_input.empty()) {
            int count = 0;
            for (auto it = model_input.begin(); it != model_input.end(); ++it) {
                native_log_stream << " " << *it;
            }
        }
        native_log_stream << std::endl;
    }

    /// outputs
    if (output_logging) {
        native_log_stream << "Output:";
        if (!model_output.empty()) {
            int count = 0;
            for (auto it = model_output.begin(); it != model_output.end(); ++it) {
                native_log_stream << " " << *it;
            }
        }
        native_log_stream << std::endl;
    }


    /// input image
    if (!playback && inputImageUpdated && !inputImage.empty()) {
//    if (inputImageUpdated && !inputImage.empty()) {
//        __android_log_print(ANDROID_LOG_ERROR, TAG, inputFile.c_str());
        cv::Mat logImage;
        cv::cvtColor(inputImage, logImage, cv::COLOR_RGB2BGR);
        cv::imwrite(inputFile, logImage);
        inputImageUpdated = false;
    }
    // logging end

    frameID++;
    logFile = logDir + frameID_to_imageName(frameID);
    inputFile = inputDir + frameID_to_imageName(frameID) + ".png";

    if (per_layer_logging) {
        m_Benchmark->listeners_.OnBenchmarkEnd({model_size_mb, startup_latency_us, input_bytes,
                                                run_stats, run_stats, init_mem_usage,
                                                overall_mem_usage});
        m_Benchmark->listeners_.OnBenchmarkStart(m_Benchmark->params_);
    }

    frame_end_us = profiling::time::NowMicros();
    frame_latency_us = frame_end_us - frame_start_us;
    native_log_stream << "Frame Start time: " << frame_start_us / 1e3 << "ms" << std::endl;
    native_log_stream << "FPS Latency: " << frame_latency_us / 1e3 << "ms" << std::endl;

    native_log_stream.close();
}