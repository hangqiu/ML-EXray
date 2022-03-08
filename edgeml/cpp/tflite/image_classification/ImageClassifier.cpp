#include <android/log.h>
#include "ImageClassifier.h"
#include <opencv2/imgproc.hpp>
//#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
//#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/model.h"

using namespace cv;

ImageClassifier::ImageClassifier(const char *tfliteModel,
                                 string nativeLogDir,
                                 long modelSize,
                                 bool playback_, bool quantized,
                                 int model_input_size,
                                 string acceleartor_choice,
                                 int numThreads,
                                 string logging_choice,
                                 string resizing_func_choice,
                                 string scale_range_choice,
                                 string channel_choice
) {
    /// log the config
    __android_log_print(ANDROID_LOG_ERROR, TAG, "Configuration in CPP:");
    __android_log_print(ANDROID_LOG_ERROR, TAG, ("Accelerator: " + acceleartor_choice).c_str());
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        ("Num Threads: " + std::to_string(numThreads)).c_str());
    __android_log_print(ANDROID_LOG_ERROR, TAG, ("Logging Level: " + logging_choice).c_str());
    __android_log_print(ANDROID_LOG_ERROR, TAG,
                        ("Resising Function: " + resizing_func_choice).c_str());
    __android_log_print(ANDROID_LOG_ERROR, TAG, ("Input Scale: " + scale_range_choice).c_str());
    __android_log_print(ANDROID_LOG_ERROR, TAG, ("Channel: " + channel_choice).c_str());

    m_modelQuantized = quantized;
    m_numThreads = numThreads;
    DETECTION_MODEL_SIZE = model_input_size;
    m_resizing_func_choice = resizing_func_choice;
    m_channel_choice = channel_choice;
    m_input_scale_range = scale_range_choice;
    //////////////////////////////////////////
    if (acceleartor_choice == "GPU") {
        use_GPUDelegate = true;
    }
    //////////////////////////////////////////

    if (logging_choice=="NONE"){
        no_logging_ = true;
    }
    if (logging_choice == "IO") {
        output_logging_ = true;
//        input_logging_ = true;
    }
    if (logging_choice == "EMBEDDING") {
        input_logging_ = true;
        output_logging_ = true;
        embedding_logging_ = true;
    }
    if (logging_choice == "PERLAYER") {
        input_logging_ = true;
        output_logging_ = true;
        embedding_logging_ = true;
        per_layer_logging_ = true;
    }
    mlMonitorNative = new EdgeMLMonitor_Native(nativeLogDir, playback_,
                                               no_logging_,
                                               input_logging_,
                                               output_logging_,
                                               embedding_logging_,
                                               per_layer_logging_);
    //////////////////////////////////////////
    if (modelSize > 0) {
        mlMonitorNative->
                onTFliteInitStart(modelSize);
        initClassificationModel(tfliteModel, modelSize
        );
        if (per_layer_logging_) {
            mlMonitorNative->initTFLiteBenchmarkModel(&m_interpreter);
        }
        mlMonitorNative->onTFliteInitStop(&m_interpreter);
    }
}

ImageClassifier::~ImageClassifier() {
    if (per_layer_logging_) {
        mlMonitorNative->endTFLiteBenchmarkModel();
    }
    free(mlMonitorNative);
    if (m_modelBytes != nullptr) {
        free(m_modelBytes);
        m_modelBytes = nullptr;
    }
    m_hasDetectionModel = false;

//    if (use_GPUDelegate) {
//        TfLiteGpuDelegateV2Delete(m_delegate);
//    }
}

// Credit: https://github.com/YijinLiu/tf-cpu/blob/master/benchmark/obj_detect_lite.cc
void ImageClassifier::initClassificationModel(const char *tfliteModel, long modelSize) {
    if (modelSize < 1) { return; }

    // Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
    m_modelBytes = (char *) malloc(sizeof(char) * modelSize);
    memcpy(m_modelBytes, tfliteModel, sizeof(char) * modelSize);
    m_model = tflite::FlatBufferModel::BuildFromBuffer(m_modelBytes, modelSize);

    if (m_model == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to load model");
        return;
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
//	tflite::ops::builtin::BuiltinRefOpResolver resolver;
    tflite::InterpreterBuilder builder(*m_model, resolver);
    if (per_layer_logging_) {
        builder.PreserveAllTensorsExperimental();
    }
    builder(&m_interpreter);
    if (m_interpreter == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to create interpreter");
        return;
    }

//    // Prepare GPU delegate.
//    if (use_GPUDelegate) {
//        m_delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
//        if (m_interpreter->ModifyGraphWithDelegate(m_delegate) != kTfLiteOk) {
//            __android_log_print(ANDROID_LOG_ERROR, TAG,
//                                "Failed to modify graph with gpu delegate!");
//            return;
//        }
//    }

    // Allocate tensor buffers.
    if (m_interpreter->AllocateTensors() != kTfLiteOk) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to allocate tensors!");
        return;
    }

    m_interpreter->SetNumThreads(m_numThreads);

    // Find input tensors.
    if (m_interpreter->inputs().size() != 1) {
        __android_log_print(ANDROID_LOG_ERROR, TAG,
                            "Classification model graph needs to have 1 and only 1 input!");
        return;
    }

    m_input_tensor = m_interpreter->tensor(m_interpreter->inputs()[0]);
    if (m_modelQuantized && m_input_tensor->type != kTfLiteUInt8) {
        __android_log_print(ANDROID_LOG_ERROR, TAG,
                            "Classification model input should be kTfLiteUInt8!");
        return;
    }

    if (!m_modelQuantized && m_input_tensor->type != kTfLiteFloat32) {
        __android_log_print(ANDROID_LOG_ERROR, TAG,
                            "Classification model input should be kTfLiteFloat32!");
        return;
    }

    if (m_input_tensor->dims->data[0] != 1 ||
        m_input_tensor->dims->data[1] != DETECTION_MODEL_SIZE ||
        m_input_tensor->dims->data[2] != DETECTION_MODEL_SIZE ||
        m_input_tensor->dims->data[3] != DETECTION_MODEL_CNLS) {
        __android_log_print(ANDROID_LOG_ERROR, TAG,
                            "Classification model must have input dims of 1x%ix%ix%i",
                            DETECTION_MODEL_SIZE,
                            DETECTION_MODEL_SIZE, DETECTION_MODEL_CNLS);
        return;
    }

    // Find output tensors.
    if (m_interpreter->outputs().size() != 1) {
        __android_log_print(ANDROID_LOG_ERROR, TAG,
                            "Classification model graph needs to have 1 and only 1 outputs!");
        return;
    }

    m_output_scores = m_interpreter->tensor(m_interpreter->outputs()[0]);

    // add intermediate layer outputs
    // visualize all tensors
//	string embedding_layer_name = "MobilenetV2/Logits/AvgPool";
//	string embedding_layer_name = "mobilenetv2_1.00_224/global_average_pooling2d/Mean";
    string embedding_layer_name = "";
    int tensor_size = m_interpreter->tensors_size();
    __android_log_print(ANDROID_LOG_ERROR, TAG, "Model tensor size %d", tensor_size);
    for (int j = 0; j < tensor_size; j++) {
        // last tensor is null tensor
        if (!m_interpreter->tensor(j)->name)continue;
        __android_log_print(ANDROID_LOG_ERROR, TAG, "%d:%s", j, m_interpreter->tensor(j)->name);
        if (m_interpreter->tensor(j)->name == embedding_layer_name) {
            m_last_tensor = m_interpreter->tensor(j);
            TfLiteIntArray *last_tensor_dim = m_last_tensor->dims;
            for (int i = 0; i < last_tensor_dim->size; i++) {
                __android_log_print(ANDROID_LOG_ERROR, TAG, "%d", last_tensor_dim->data[i]);
            }
        }
    }
    m_hasDetectionModel = true;
    __android_log_print(ANDROID_LOG_ERROR, TAG, "Finished tflite Initialization");
}

ClassificationResult *ImageClassifier::classify(Mat src) {
    mlMonitorNative->startCurrentFrame();

    ClassificationResult res[DETECT_NUM];
    if (!m_hasDetectionModel) {
        return res;
    }

//    mlMonitorNative->onRandomOpsStart("PreprocessingTotal");
//    mlMonitorNative->onRandomOpsStart("Resizing");
    Mat image;
    if (m_resizing_func_choice == "AVG_AREA") {
        resize(src, image, Size(DETECTION_MODEL_SIZE, DETECTION_MODEL_SIZE), 0, 0, INTER_AREA);
    } else if (m_resizing_func_choice == "BILINEAR") {

        resize(src, image, Size(DETECTION_MODEL_SIZE, DETECTION_MODEL_SIZE), 0, 0, INTER_LINEAR);
    } else {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Resizing function not supported");
    }
//    mlMonitorNative->onRandomOpsStop("Resizing");
//    mlMonitorNative->onRandomOpsStart("MemCopy");
    int cnls = image.type();
    if (cnls == CV_8UC1) {
        cvtColor(image, image, COLOR_GRAY2RGB);
    } else if (cnls == CV_8UC4) {
        cvtColor(image, image, COLOR_BGRA2RGB);
    } else {
        // for CV_8UC3 assume RGB output
    }
    if (m_channel_choice == "BGR") {
        cvtColor(image, image, COLOR_RGB2BGR);
    }
    if (m_modelQuantized) {
        // Copy image into input tensor
        uint8_t *dst = m_input_tensor->data.uint8;
        memcpy(dst, image.data,
               sizeof(uint8_t) * DETECTION_MODEL_SIZE * DETECTION_MODEL_SIZE *
               DETECTION_MODEL_CNLS);
    } else {
        // Normalize the image based on std and mean (p' = (p-mean)/std)
        // this is scaling to [-1, 1]
        Mat fimage;
        if (m_input_scale_range == "MINUSONE_ONE") {
            image.convertTo(fimage, CV_32FC3, 1 / IMAGE_STD, -IMAGE_MEAN / IMAGE_STD);
        } else if (m_input_scale_range == "ZERO_ONE") {
            image.convertTo(fimage, CV_32FC3, 1 / 255.0, 0);
        } else {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "Input scale range not supported");
        }

        // Copy image into input tensor
        float *dst = m_input_tensor->data.f;
        memcpy(dst, fimage.data,
               sizeof(float) * DETECTION_MODEL_SIZE * DETECTION_MODEL_SIZE * DETECTION_MODEL_CNLS);
    }
//    mlMonitorNative->onRandomOpsStop("MemCopy");
//    mlMonitorNative->onRandomOpsStop("PreprocessingTotal");

    mlMonitorNative->onTFliteInferenceStart(image);
    TfLiteStatus status = m_interpreter->Invoke();
    mlMonitorNative->onTFliteInferenceStop(&m_interpreter, m_input_tensor, m_last_tensor,
                                           m_output_scores, m_modelQuantized);
    if (status != kTfLiteOk) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Error invoking detection model");
        return res;
    }

    const void *detection_scores = nullptr;
    if (m_modelQuantized) {
        detection_scores = m_output_scores->data.uint8;
    } else {
        detection_scores = m_output_scores->data.f;
    }

    vector<ClassificationResult> res_vec;
    for (int i = 0; i < NUM_CLASS; ++i) {
        ClassificationResult tmp;
        tmp.score = m_modelQuantized ? ((uint8_t *) detection_scores)[i] / 255.0
                                     : ((float *) detection_scores)[i];
        tmp.label = i;
        res_vec.push_back(tmp);
    }
    sort(res_vec.begin(), res_vec.end(), greater<ClassificationResult>());

    for (int j = 0; j < DETECT_NUM; j++) {
        res[j].label = res_vec[j].label;
        res[j].score = res_vec[j].score;
    }

    mlMonitorNative->endCurrentFrame();
    return res;
}
