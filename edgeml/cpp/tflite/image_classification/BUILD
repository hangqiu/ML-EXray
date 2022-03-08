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

cc_library(
    name = "tflite_image_classifier_lib",
    srcs = ["ImageClassifier.cpp"],
    hdrs = ["ImageClassifier.h"],
    copts = common_copts + select({
        "@org_tensorflow//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//edgeml/cpp/tflite/edgeml_monitor_native:edgeml_monitor_native_lib",
        "//third_party:opencv_binary",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/c:common",
        #        "@org_tensorflow//tensorflow/lite/delegates/gpu:delegate",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels:reference_ops",
    ],
)
