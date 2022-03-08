//
// Created by hang on 1/13/22.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "edgeml/cpp/Model_Runner/TfliteImageClassificationModelRunner.h"

using namespace cv;

int main() {
    const char *model_path = "model/ConvertedModels/MobileNetV2/MobileNetV2_imagenet_224.tflite";
    std::string image_path = "data/0_data/imagenet2012_1/nativeInput/ILSVRC2012_val_00000001.JPEG";

    TfliteImageClassificationModelRunner m_runner = TfliteImageClassificationModelRunner(
            224,
            false);
    m_runner.init(model_path);
    Mat input = imread(image_path);
    cv::imshow("",input);
    std::vector<ClassificationResult> res = m_runner.invoke_model(input);

    for (ClassificationResult result : res) {
        std::cout << result.label << "," << result.score << std::endl;
        break;
    }

}
