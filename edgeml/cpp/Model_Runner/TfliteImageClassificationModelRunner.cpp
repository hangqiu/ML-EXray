//
// Created by hang on 1/13/22.
//

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

#include <iostream>
#include <opencv2/imgproc.hpp>
#include "TfliteImageClassificationModelRunner.h"

using namespace cv;
using namespace std;

TfliteImageClassificationModelRunner::TfliteImageClassificationModelRunner(
        int input_size,
        bool quantized,
        bool per_layer_logging,
        std::string resizing_func_choice,
        std::string channel_choice,
        std::string input_scale_range) {

    _input_size = input_size;
    _quantized = quantized;
    _per_layer_logging = per_layer_logging;
    _resizing_func_choice = resizing_func_choice;
    _channel_choice = channel_choice;
    _input_scale_range = input_scale_range;
}


void TfliteImageClassificationModelRunner::init(const char *tflite_model_path) {
    // init mlMonitor
    mlMonitorNative = new EdgeMLMonitor_Native("./",
                                               true,
                                               false,
                                               true,
                                               true,
                                               true,
                                               true);

    // Load model
    model = tflite::FlatBufferModel::BuildFromFile(tflite_model_path);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    mlMonitorNative->
            onTFliteInitStart(0);

    // Build the interpreter with the InterpreterBuilder.
    // Note: all Interpreters should be built with the InterpreterBuilder,
    // which allocates memory for the Interpreter and does various set up
    // tasks so that the Interpreter can read the provided model.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    //	tflite::ops::builtin::BuiltinRefOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);

    if (_per_layer_logging) {
        builder.PreserveAllTensorsExperimental();
    }
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

//    // Prepare GPU delegate.
//    if (_use_GPUDelegate) {
//        TfLiteDelegate *m_delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
//        TFLITE_MINIMAL_CHECK(interpreter->ModifyGraphWithDelegate(m_delegate) != kTfLiteOk);
//    }

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
//    printf("=== Pre-invoke Interpreter State ===\n");
//    tflite::PrintInterpreterState(interpreter.get());


    // Find input tensors.
    TFLITE_MINIMAL_CHECK(interpreter->inputs().size() == 1)

    m_input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    if (_quantized) {
        TFLITE_MINIMAL_CHECK(m_input_tensor->type == kTfLiteUInt8);
    } else {
        TFLITE_MINIMAL_CHECK(m_input_tensor->type == kTfLiteFloat32);
    }

    TFLITE_MINIMAL_CHECK (m_input_tensor->dims->data[0] == 1);
    TFLITE_MINIMAL_CHECK (m_input_tensor->dims->data[1] == _input_size);
    TFLITE_MINIMAL_CHECK (m_input_tensor->dims->data[2] == _input_size);
    TFLITE_MINIMAL_CHECK (m_input_tensor->dims->data[3] == DETECTION_MODEL_CNLS);

    // Find output tensors.
    TFLITE_MINIMAL_CHECK (interpreter->outputs().size() == 1);
    m_output_scores = interpreter->tensor(interpreter->outputs()[0]);



    if (_per_layer_logging) {
        mlMonitorNative->initTFLiteBenchmarkModel(&interpreter);
    }
    mlMonitorNative->onTFliteInitStop(&interpreter);
}

std::vector<ClassificationResult>
TfliteImageClassificationModelRunner::invoke_model(cv::Mat input) {

    mlMonitorNative->startCurrentFrame();
    // Fill input buffers
    Mat image;
    if (_resizing_func_choice == "AVG_AREA") {
        resize(input, image, Size(_input_size, _input_size), 0, 0, INTER_AREA);
    } else if (_resizing_func_choice == "BILINEAR") {

        resize(input, image, Size(_input_size, _input_size), 0, 0, INTER_LINEAR);
    } else {
        std::perror("resizing function not valid");
        exit(1);
    }

    int cnls = image.type();
    if (cnls == CV_8UC1) {
        cvtColor(image, image, COLOR_GRAY2RGB);
    } else if (cnls == CV_8UC4) {
        cvtColor(image, image, COLOR_BGRA2RGB);
    } else {
        // for CV_8UC3 assume RGB output, do nothing
    }
    if (_channel_choice == "BGR") {
        cvtColor(image, image, COLOR_RGB2BGR);
    }
    if (_quantized) {
        // Copy image into input tensor
        uint8_t *dst = m_input_tensor->data.uint8;
        memcpy(dst, image.data,
               sizeof(uint8_t) * _input_size * _input_size *
               DETECTION_MODEL_CNLS);
    } else {
        // Normalize the image based on std and mean (p' = (p-mean)/std)
        // this is scaling to [-1, 1]
        Mat fimage;
        if (_input_scale_range == "MINUSONE_ONE") {
            image.convertTo(fimage, CV_32FC3, 1 / IMAGE_STD, -IMAGE_MEAN / IMAGE_STD);
        } else if (_input_scale_range == "ZERO_ONE") {
            image.convertTo(fimage, CV_32FC3, 1 / 255.0, 0);
        } else {
            std::perror("normalization function not valid");
            exit(1);
        }

        // Copy image into input tensor
        float *dst = m_input_tensor->data.f;
        memcpy(dst, fimage.data,
               sizeof(float) * _input_size * _input_size * DETECTION_MODEL_CNLS);
    }

    // Run inference
    mlMonitorNative->onTFliteInferenceStart(image);
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    mlMonitorNative->onTFliteInferenceStop(&interpreter, m_input_tensor, m_last_tensor,
                                           m_output_scores, _quantized);

//    printf("\n\n=== Post-invoke Interpreter State ===\n");
//    tflite::PrintInterpreterState(interpreter.get());

    // Read output buffers
    const void *detection_scores = nullptr;
    if (_quantized) {
        detection_scores = m_output_scores->data.uint8;
    } else {
        detection_scores = m_output_scores->data.f;
    }

    std::vector<ClassificationResult> res_vec;
    for (int i = 0; i < NUM_CLASS; ++i) {
        ClassificationResult tmp;
        tmp.score = _quantized ? ((uint8_t *) detection_scores)[i] / 255.0
                               : ((float *) detection_scores)[i];
        tmp.label = i;
        res_vec.push_back(tmp);
//        std::cout << tmp.label << "," << tmp.score << std::endl;
    }
    sort(res_vec.begin(), res_vec.end(), greater<ClassificationResult>());
    mlMonitorNative->endCurrentFrame();
    return res_vec;
}