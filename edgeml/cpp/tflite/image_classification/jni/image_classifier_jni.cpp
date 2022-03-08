#include <jni.h>
#include <string>
#include <sstream>
#include <iostream>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
//#include "mediapipe/framework/port/opencv_core_inc.h"
//#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "edgeml/cpp/tflite/image_classification/ImageClassifier.h"

using namespace cv;

#define LOG_E(...) __android_log_write(ANDROID_LOG_ERROR, "JNI_LOGS", __VA_ARGS__);

void rotateMat(cv::Mat &matImage, int rotation) {
    if (rotation == 90) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 1); //transpose+flip(1)=CW
    } else if (rotation == 270) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 0); //transpose+flip(0)=CCW
    } else if (rotation == 180) {
        flip(matImage, matImage, -1);    //flip(-1)=180
    }
}

std::string jstring2string(JNIEnv *env, jstring jStr) {
    if (!jStr)
        return "";

    const jclass stringClass = env->GetObjectClass(jStr);
    const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
    const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes,
                                                                       env->NewStringUTF("UTF-8"));

    size_t length = (size_t) env->GetArrayLength(stringJbytes);
    jbyte *pBytes = env->GetByteArrayElements(stringJbytes, NULL);

    std::string ret = std::string((char *) pBytes, length);
    env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

    env->DeleteLocalRef(stringJbytes);
    env->DeleteLocalRef(stringClass);
    return ret;
}

int parseModelInputSize(std::string model_name_str) {
    // parse model input size from model name
    std::stringstream model_name_stream(model_name_str);
    std::string input_size_str;
    while (getline(model_name_stream, input_size_str, '_')) {
        continue;
    }
    std::stringstream input_size_stream(input_size_str);
    getline(input_size_stream, input_size_str, '.');
    int input_size = stoi(input_size_str);
    return input_size;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_edgeml_edgemlinsight_NativeClassifier_initClassifier(
        JNIEnv *env,
        jobject p_this,
        jobject assetManager,
        jstring nativeLogFile,
        jstring model_name,
        jstring m_accelerator,
        int num_threads,
        jstring m_logging,
        jstring m_resizing_func,
        jstring m_scale_range,
        jstring m_channel,
        int m_rotation,
        jint playback) {
//	fprintf(stderr, "Begin");
    char *buffer = nullptr;
    long size = 0;

    std::string nativeString = jstring2string(env, nativeLogFile);
    std::string model_name_str = jstring2string(env, model_name);

    std::string m_accelerator_str = jstring2string(env, m_accelerator);
    std::string m_logging_str = jstring2string(env, m_logging);
    std::string m_resizing_func_str = jstring2string(env, m_resizing_func);
    std::string m_scale_range_str = jstring2string(env, m_scale_range);
    std::string m_channel_str = jstring2string(env, m_channel);

    bool quantized = false;
    std::string quant = "quant";
    if (model_name_str.find(quant) != std::string::npos) {
        quantized = true;
    }

    int model_input_size = parseModelInputSize(model_name_str);

    if (!(env->IsSameObject(assetManager, NULL))) {
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        AAsset *asset = AAssetManager_open(mgr, model_name_str.c_str(), AASSET_MODE_UNKNOWN);
        assert(asset != nullptr);

        size = AAsset_getLength(asset);
        buffer = (char *) malloc(sizeof(char) * size);
        AAsset_read(asset, buffer, size);
        AAsset_close(asset);
    }

    jlong obj = (jlong) new ImageClassifier(
            buffer,
            nativeString,
            size,
            playback,
            quantized,
            model_input_size,
            m_accelerator_str,
            num_threads,
            m_logging_str,
            m_resizing_func_str,
            m_scale_range_str,
            m_channel_str
    );
    free(buffer); // ObjectDetector duplicate it and responsible to free it
    return obj;
}


extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_edgeml_edgemlinsight_NativeClassifier_classify(JNIEnv *env, jobject p_this,
                                                        jlong detectorAddr, jbyteArray src,
                                                        int width, int height, int rotation) {
    // Frame bytes to Mat
    jbyte *_yuv = env->GetByteArrayElements(src, 0);
    Mat myyuv(height + height / 2, width, CV_8UC1, _yuv);
    Mat frame(height, width, CV_8UC4);
    cvtColor(myyuv, frame, COLOR_YUV2BGRA_NV21); // seems already in RGBA format!!!
    rotateMat(frame, rotation);
    // frame = frame(Rect(0, 0, frame.cols, frame.cols));
    env->ReleaseByteArrayElements(src, _yuv, 0);

    // Detect
    ImageClassifier *detector = (ImageClassifier *) detectorAddr;
    ClassificationResult *res = detector->classify(frame);

    // Encode each detection as 6 numbers (label,score,xmin,xmax,ymin,ymax)
    int resArrLen = detector->DETECT_NUM * 2;
    jfloat jres[resArrLen];
    for (int i = 0; i < detector->DETECT_NUM; ++i) {
        jres[i * 2] = res[i].label;
        jres[i * 2 + 1] = res[i].score;
    }

    jfloatArray detections = env->NewFloatArray(resArrLen);
    env->SetFloatArrayRegion(detections, 0, resArrLen, jres);

    return detections;
}



extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_edgeml_edgemlinsight_NativeClassifier_playbackclassify(JNIEnv *env, jobject p_this,
                                                                jlong detectorAddr,
                                                                jobject obj_bitmap, int rotation) {
    // Frame bytes (read from png) to Mat
    cv::Mat matrix;
    void *bitmapPixels;                                            // Save picture pixel data
    AndroidBitmapInfo bitmapInfo;                                   // Save picture parameters

    AndroidBitmap_getInfo(env, obj_bitmap, &bitmapInfo);        // Get picture parameters
    AndroidBitmap_lockPixels(env, obj_bitmap,
                             &bitmapPixels);  // Get picture pixels (lock memory block)


    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        __android_log_print(ANDROID_LOG_ERROR, "jni", "ANDROID_BITMAP_FORMAT_RGBA_8888");
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4,
                    bitmapPixels);    // Establish temporary mat
        cv::cvtColor(tmp, matrix,
                     cv::COLOR_RGBA2BGRA);                                                        // Copy to target matrix
    } else {
        __android_log_print(ANDROID_LOG_ERROR, "jni", "cv8uc2");
        cv::Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cv::cvtColor(tmp, matrix, cv::COLOR_BGR5652RGB);
    }

    AndroidBitmap_unlockPixels(env, obj_bitmap);            // Unlock

    rotateMat(matrix, rotation);

    // Detect
    ImageClassifier *detector = (ImageClassifier *) detectorAddr;
    ClassificationResult *res = detector->classify(matrix);

    // Encode each detection as 6 numbers (label,score,xmin,xmax,ymin,ymax)
    int resArrLen = detector->DETECT_NUM * 2;
    jfloat jres[resArrLen];
    for (int i = 0; i < detector->DETECT_NUM; ++i) {
        jres[i * 2] = res[i].label;
        jres[i * 2 + 1] = res[i].score;
    }

    jfloatArray detections = env->NewFloatArray(resArrLen);
    env->SetFloatArrayRegion(detections, 0, resArrLen, jres);

    return detections;
}

