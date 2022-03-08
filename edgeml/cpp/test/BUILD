load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_android//android:rules.bzl", "android_binary", "android_library")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")

common_copts = tflite_copts() + tflite_copts_warnings()

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_binary(
    name = "tflite_image_classifier_test",
    srcs = ["ImageClassifierTest.cpp"],
    deps = [
        "//edgeml/cpp/Model_Runner:tflite_image_classifier_runner",
        "//third_party:opencv_binary",
    ],
)
